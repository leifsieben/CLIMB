"""v2 data loaders.

Three components:
- MLM streaming: reuses StreamingTokenizedDataset from data.py (already worker-shard correct).
- Supervised in-RAM: reads the supervised wide tokenized parquet for the requested
  families' label columns into a single tensor pair (input_ids, labels). Replaces the
  buggy v1 StreamingSupervisedFamilyDataset.
- A simple iterator that yields per-batch mode flags (MLM vs supervised) at the
  configured ratio so the trainer can switch heads.

The supervised label tensor uses NaN for missing labels; the loss ignores NaN entries.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from data import StreamingTokenizedDataset, _stable_fraction_membership
from storage_utils import is_s3_uri, materialize_path, parquet_dataset


# ---------- supervised in-RAM loader ----------

def _family_columns(schema, families: List[str]) -> List[str]:
    """Pick *numeric* label columns belonging to the given families. Convention:
    column names start with `<family>__`. We always exclude `input_ids` and
    `attention_mask`, plus any column whose pyarrow type is not numeric.

    `schema` may be either a list of column names (legacy) or a pyarrow.Schema.
    """
    import pyarrow as pa
    is_schema = hasattr(schema, "field")
    schema_columns = schema.names if is_schema else list(schema)

    EXCLUDE = {"input_ids", "attention_mask", "smiles", "SMILES", "canonical_smiles", "canonical_SMILES"}

    def _is_numeric(name: str) -> bool:
        if not is_schema:
            return True  # legacy path: caller is responsible
        field = schema.field(name)
        t = field.type
        return pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t)

    out = []
    for col in schema_columns:
        if col in EXCLUDE:
            continue
        for fam in families:
            if col.startswith(f"{fam}__") or col == fam:
                if _is_numeric(col):
                    out.append(col)
                break
    return out


def load_supervised_inram(
    parquet_path: str,
    families: List[str],
    max_length: int = 512,
    max_rows: Optional[int] = None,
    min_label_density: float = 0.001,
    label_dtype: str = "float16",
    default_max_rows: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Load (input_ids, attention_mask, labels) into RAM for the requested families.

    Memory-conscious: pre-allocates output tensors based on parquet row count, writes
    directly into them (no chunk-list-then-concat). Drops very sparse label columns
    (default <0.1% non-NaN) which contribute essentially zero gradient signal but
    consume real memory. Stores labels as float16 (precision ~1e-3, fine for z-scored
    targets with magnitude <10).

    Returns:
        input_ids: int64 tensor [N, max_length]
        attention_mask: int64 tensor [N, max_length]
        labels: float16 (or as configured) tensor [N, T]; NaN for missing
        column_names: ordered list of label columns kept after density filter
        task_types: per-column "classification" or "regression"
    """
    import pyarrow.dataset as pads

    if is_s3_uri(parquet_path):
        ds_obj = parquet_dataset(parquet_path)
    else:
        ds_obj = pads.dataset(parquet_path, format="parquet")

    schema = ds_obj.schema
    schema_cols = schema.names
    label_cols = _family_columns(schema, families)
    if not label_cols:
        raise ValueError(
            f"No label columns matched families {families} in parquet {parquet_path}"
        )
    if "input_ids" not in schema_cols or "attention_mask" not in schema_cols:
        raise ValueError(f"Parquet {parquet_path} missing input_ids/attention_mask columns")

    # Get total row count to pre-allocate (avoid chunk-list growth).
    total_rows = ds_obj.count_rows()
    cap = max_rows if max_rows is not None else default_max_rows
    n_alloc = min(total_rows, cap)
    np_dtype = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32}[label_dtype]
    label_bytes = 2 if label_dtype == "float16" else 4
    expected_label_gb = n_alloc * len(label_cols) * label_bytes / 1e9
    print(
        f"[load_supervised_inram] {len(label_cols)} label cols, {n_alloc}/{total_rows} rows, "
        f"~{expected_label_gb:.1f} GB labels in RAM (dtype={label_dtype})",
        flush=True,
    )

    columns = ["input_ids", "attention_mask"] + label_cols

    # Pre-allocate (writes directly avoid concat double-copy).
    input_ids = np.zeros((n_alloc, max_length), dtype=np.int32)
    attention_mask = np.zeros((n_alloc, max_length), dtype=np.int8)
    labels = np.full((n_alloc, len(label_cols)), np.nan, dtype=np_dtype)

    write_idx = 0
    rows_scanned = 0
    for batch_idx, batch in enumerate(ds_obj.to_batches(columns=columns, batch_size=8192)):
        ids_arrow = batch.column(batch.schema.get_field_index("input_ids"))
        mask_arrow = batch.column(batch.schema.get_field_index("attention_mask"))
        batch_n = len(ids_arrow)
        rows_scanned += batch_n
        if write_idx >= n_alloc:
            break

        # Pull labels for this batch first; identify rows with at least one non-NaN label.
        batch_labels = np.full((batch_n, len(label_cols)), np.nan, dtype=np_dtype)
        for j, col in enumerate(label_cols):
            arr = batch.column(batch.schema.get_field_index(col))
            np_col = arr.to_numpy(zero_copy_only=False)
            if np_col.dtype == np.object_:
                col_floats = np.array(
                    [float(v) if v is not None else np.nan for v in np_col],
                    dtype=np_dtype,
                )
            else:
                col_floats = np_col.astype(np_dtype, copy=False)
            batch_labels[:, j] = col_floats

        # Filter: only keep rows with at least one finite label.
        # For float16/float32, use np.isfinite to also exclude inf values upfront.
        if np_dtype == np.float16:
            # np.isfinite supports float16
            row_keep = np.isfinite(batch_labels).any(axis=1)
        else:
            row_keep = np.isfinite(batch_labels).any(axis=1)

        kept_n = int(row_keep.sum())
        if kept_n == 0:
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}: scanned {rows_scanned}, kept {write_idx} (0 in this batch)", flush=True)
            continue

        # Cap kept rows so we don't overflow the alloc.
        if write_idx + kept_n > n_alloc:
            # Take only as many kept rows as fit.
            keep_indices = np.where(row_keep)[0][: n_alloc - write_idx]
            row_keep = np.zeros_like(row_keep)
            row_keep[keep_indices] = True
            kept_n = int(row_keep.sum())

        # Pad input_ids / mask for kept rows only.
        ids_flat = np.asarray(ids_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int32)
        mask_flat = np.asarray(mask_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int8)
        ids_offsets = ids_arrow.offsets.to_numpy(zero_copy_only=False)
        mask_offsets = mask_arrow.offsets.to_numpy(zero_copy_only=False)

        kept_indices = np.where(row_keep)[0]
        for k, i in enumerate(kept_indices):
            s_id, e_id = int(ids_offsets[i]), int(ids_offsets[i + 1])
            s_m, e_m = int(mask_offsets[i]), int(mask_offsets[i + 1])
            L_id = min(e_id - s_id, max_length)
            L_m = min(e_m - s_m, max_length)
            if L_id > 0:
                input_ids[write_idx + k, :L_id] = ids_flat[s_id : s_id + L_id]
            if L_m > 0:
                attention_mask[write_idx + k, :L_m] = mask_flat[s_m : s_m + L_m]

        labels[write_idx : write_idx + kept_n] = batch_labels[row_keep]
        write_idx += kept_n

        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}: scanned {rows_scanned}, kept {write_idx}/{n_alloc}", flush=True)

    # Trim if early-stopped
    input_ids = input_ids[:write_idx]
    attention_mask = attention_mask[:write_idx]
    labels = labels[:write_idx]

    # Drop label columns that are too sparse to be useful.
    if min_label_density > 0:
        density = (~np.isnan(labels)).mean(axis=0)
        keep_cols = density >= min_label_density
        if not keep_cols.all():
            kept = int(keep_cols.sum())
            dropped = int((~keep_cols).sum())
            print(f"  dropping {dropped} cols with <{min_label_density:.1%} density; keeping {kept}", flush=True)
            labels = labels[:, keep_cols]
            label_cols = [c for c, k in zip(label_cols, keep_cols) if k]

    # Drop rows with no labels at all (no signal).
    has_label = (~np.isnan(labels)).any(axis=1)
    n_kept = int(has_label.sum())
    if n_kept < write_idx:
        print(f"  dropping {write_idx - n_kept} all-NaN rows; keeping {n_kept}", flush=True)
        input_ids = input_ids[has_label]
        attention_mask = attention_mask[has_label]
        labels = labels[has_label]

    # Convert to torch.
    input_ids_t = torch.from_numpy(input_ids).to(torch.int64)
    attention_mask_t = torch.from_numpy(attention_mask).to(torch.int64)
    labels_t = torch.from_numpy(labels)
    del input_ids, attention_mask, labels
    labels = labels_t
    input_ids = input_ids_t
    attention_mask = attention_mask_t

    # Infer task type per column AND z-score normalise regression columns to keep
    # the supervised loss in the same magnitude as the MLM loss. Without this,
    # raw L1000 columns (gene-expression z-scores, magnitude up to ~1000) blow up
    # the gradient and the encoder learns only the supervised signal.
    task_types: List[str] = []
    for j in range(labels.shape[1]):
        # Compute stats in float32 for stability, write back in the storage dtype.
        col_f32 = labels[:, j].to(torch.float32)
        # Treat both NaN and inf as missing — inf in raw data would propagate
        # through z-score normalization and produce NaN losses downstream.
        valid = torch.isfinite(col_f32)
        if valid.sum() == 0:
            task_types.append("regression")
            # Replace any non-finite values with NaN before storing.
            col_f32 = torch.where(valid, col_f32, torch.tensor(float("nan")))
            labels[:, j] = col_f32.to(labels.dtype)
            continue
        unique_vals = torch.unique(col_f32[valid])
        if unique_vals.numel() <= 2 and torch.all((unique_vals == 0) | (unique_vals == 1)):
            task_types.append("classification")
            # Mark non-finite as NaN; classification keeps {0,1} values otherwise.
            col_f32 = torch.where(valid, col_f32, torch.tensor(float("nan")))
            labels[:, j] = col_f32.to(labels.dtype)
            continue
        task_types.append("regression")
        mean = col_f32[valid].mean()
        std = col_f32[valid].std()
        if torch.isfinite(std) and std > 1e-6:
            col_norm = (col_f32 - mean) / std
            col_norm = torch.clamp(col_norm, min=-10.0, max=10.0)
            # Set any non-finite (inf or NaN) entries to NaN.
            col_norm = torch.where(valid, col_norm, torch.tensor(float("nan")))
            labels[:, j] = col_norm.to(labels.dtype)
        else:
            # Constant column — write zeros for valid, NaN for invalid.
            col_norm = torch.where(valid, torch.zeros_like(col_f32), torch.tensor(float("nan")))
            labels[:, j] = col_norm.to(labels.dtype)

    return input_ids, attention_mask, labels, label_cols, task_types


def load_supervised_inram_stratified(
    parquet_path: str,
    families: List[str],
    family_caps: Dict[str, int],
    max_length: int = 128,
    min_label_density: float = 0.001,
    regression_loss: str = "mae",
    label_dtype: str = "float16",
    blocklist: "Optional[set]" = None,
):
    """Family-balanced supervised loader — the fix for the PCQM-collapse bug.

    Rows are routed to a family bucket to fill a per-family cap so every requested
    family is represented. Two fixes vs the first version:
      - Routing is SCARCE-FAMILY-FIRST (smallest cap first). The parquet has multi-family
        rows (a molecule can carry both PCBA and L1000 labels); PCBA-first routing stole
        L1000's rows (→ only ~1.6k L1000 loaded). Filling small families first recovers them.
      - Optional `blocklist` (set of canonical `smiles_canon` strings) drops leaked
        eval-test molecules from the supervised training set.
    Returns an extra ``col_family`` list (family per label column) for the multi-head loss.
    """
    import pyarrow.dataset as pads

    ds_obj = parquet_dataset(parquet_path) if is_s3_uri(parquet_path) else pads.dataset(parquet_path, format="parquet")
    schema = ds_obj.schema
    if "input_ids" not in schema.names or "attention_mask" not in schema.names:
        raise ValueError(f"Parquet {parquet_path} missing input_ids/attention_mask")

    # Ordered label columns + per-column family, per requested family.
    fam_cols: Dict[str, List[str]] = {f: _family_columns(schema, [f]) for f in families}
    label_cols: List[str] = []
    col_family: List[str] = []
    fam_slice: Dict[str, slice] = {}
    for f in families:
        start = len(label_cols)
        label_cols.extend(fam_cols[f])
        col_family.extend([f] * len(fam_cols[f]))
        fam_slice[f] = slice(start, len(label_cols))
    if not label_cols:
        raise ValueError(f"No label columns matched families {families}")

    np_dtype = {"float16": np.float16, "float32": np.float32}[label_dtype]
    n_alloc = sum(int(family_caps.get(f, 100_000)) for f in families)
    input_ids = np.zeros((n_alloc, max_length), dtype=np.int32)
    attention_mask = np.zeros((n_alloc, max_length), dtype=np.int8)
    labels = np.full((n_alloc, len(label_cols)), np.nan, dtype=np_dtype)

    kept = {f: 0 for f in families}
    write_idx = 0
    have_smiles = "smiles_canon" in schema.names
    columns = ["input_ids", "attention_mask"] + label_cols + (["smiles_canon"] if (blocklist and have_smiles) else [])
    # Scarce-family-first routing order (smallest cap first) so multi-family rows are
    # claimed by the small families before the huge ones (PCBA) grab them.
    route_order = sorted(families, key=lambda f: int(family_caps.get(f, 100_000)))
    n_blocked = 0
    print(f"[stratified] families={families} route_order={route_order} "
          f"caps={ {f:family_caps.get(f) for f in families} } cols={len(label_cols)} "
          f"blocklist={len(blocklist) if blocklist else 0}", flush=True)

    for batch in ds_obj.to_batches(columns=columns, batch_size=8192):
        if all(kept[f] >= int(family_caps.get(f, 0)) for f in families):
            break
        n = batch.num_rows
        # pull labels for this batch
        bl = np.full((n, len(label_cols)), np.nan, dtype=np.float64)
        for j, c in enumerate(label_cols):
            arr = batch.column(batch.schema.get_field_index(c)).to_numpy(zero_copy_only=False)
            bl[:, j] = np.array([float(v) if v is not None else np.nan for v in arr], dtype=np.float64) \
                if arr.dtype == np.object_ else arr.astype(np.float64, copy=False)
        # drop leaked eval molecules (direct smiles_canon string match)
        blocked = np.zeros(n, dtype=bool)
        if blocklist and have_smiles:
            sc = batch.column(batch.schema.get_field_index("smiles_canon")).to_pylist()
            blocked = np.array([(s in blocklist) for s in sc], dtype=bool)
            n_blocked += int(blocked.sum())
        # route each row to a family, SCARCE-FAMILY-FIRST (fixes L1000 being stolen by PCBA)
        fam_index = {f: i for i, f in enumerate(families)}
        assigned = np.full(n, -1, dtype=np.int64)
        for f in route_order:
            has = np.isfinite(bl[:, fam_slice[f]]).any(axis=1) & (assigned < 0) & (~blocked)
            assigned[has] = fam_index[f]
        if not (assigned >= 0).any():
            continue

        ids_arrow = batch.column(batch.schema.get_field_index("input_ids"))
        mask_arrow = batch.column(batch.schema.get_field_index("attention_mask"))
        ids_flat = np.asarray(ids_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int32)
        mask_flat = np.asarray(mask_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int8)
        ids_off = ids_arrow.offsets.to_numpy(zero_copy_only=False)
        mask_off = mask_arrow.offsets.to_numpy(zero_copy_only=False)

        for fi, f in enumerate(families):
            room = int(family_caps.get(f, 0)) - kept[f]
            if room <= 0:
                continue
            rows = np.where(assigned == fi)[0][:room]
            for r in rows:
                s_id, e_id = int(ids_off[r]), int(ids_off[r + 1])
                s_m, e_m = int(mask_off[r]), int(mask_off[r + 1])
                L_id = min(e_id - s_id, max_length); L_m = min(e_m - s_m, max_length)
                if L_id > 0:
                    input_ids[write_idx, :L_id] = ids_flat[s_id:s_id + L_id]
                if L_m > 0:
                    attention_mask[write_idx, :L_m] = mask_flat[s_m:s_m + L_m]
                labels[write_idx] = bl[r].astype(np_dtype)
                write_idx += 1
            kept[f] += len(rows)

    print(f"[stratified] kept per family: {kept} (total {write_idx}, blocked {n_blocked} leaked)", flush=True)
    input_ids = input_ids[:write_idx]; attention_mask = attention_mask[:write_idx]; labels = labels[:write_idx]

    # density filter (within kept rows) — keep col_family in sync
    if min_label_density > 0 and write_idx > 0:
        density = (~np.isnan(labels)).mean(axis=0)
        keep = density >= min_label_density
        if not keep.all():
            labels = labels[:, keep]
            label_cols = [c for c, k in zip(label_cols, keep) if k]
            col_family = [cf for cf, k in zip(col_family, keep) if k]

    input_ids_t = torch.from_numpy(input_ids).to(torch.int64)
    attention_mask_t = torch.from_numpy(attention_mask).to(torch.int64)
    labels_t = torch.from_numpy(labels)

    # infer task type + normalize regression columns (z-score); MAE/MSE handled in loss
    task_types: List[str] = []
    for j in range(labels_t.shape[1]):
        col = labels_t[:, j].to(torch.float32)
        valid = torch.isfinite(col)
        if valid.sum() == 0:
            task_types.append("regression"); continue
        uniq = torch.unique(col[valid])
        if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
            task_types.append("classification")
            labels_t[:, j] = torch.where(valid, col, torch.tensor(float("nan"))).to(labels_t.dtype)
            continue
        task_types.append("regression")
        mean, std = col[valid].mean(), col[valid].std()
        if torch.isfinite(std) and std > 1e-6:
            z = torch.clamp((col - mean) / std, -10.0, 10.0)
            labels_t[:, j] = torch.where(valid, z, torch.tensor(float("nan"))).to(labels_t.dtype)
        else:
            labels_t[:, j] = torch.where(valid, torch.zeros_like(col), torch.tensor(float("nan"))).to(labels_t.dtype)

    return input_ids_t, attention_mask_t, labels_t, label_cols, task_types, col_family


class SupervisedInRAMDataset(torch.utils.data.Dataset):
    """Map-style dataset over the in-RAM supervised tensors."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_types: List[str],
    ):
        assert input_ids.shape[0] == attention_mask.shape[0] == labels.shape[0]
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.task_types = task_types

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------- MLM streaming wrapper ----------

def make_mlm_dataset(
    paths: List[str],
    max_samples: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    subset_seed: int = 0,
    cache_remote_files: bool = True,
) -> StreamingTokenizedDataset:
    """Thin wrapper. Reuses data.py:StreamingTokenizedDataset which already shards
    correctly across DataLoader workers (data.py:232).
    """
    return StreamingTokenizedDataset(
        paths,
        with_labels=False,
        shuffle=True,
        max_samples=max_samples,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        cache_remote_files=cache_remote_files,
    )


# ---------- shared MLM masking ----------

def mlm_mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    special_tokens: set,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BERT-style 80/10/10 masking. Returns (input_ids_masked, labels) where labels
    are -100 on unmasked positions. Shared by MLMCollator (pre-tokenized) and
    RawSmilesMLMCollator (raw-SMILES + on-the-fly tokenization) so both paths mask
    identically.
    """
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_probability)

    special_mask = (attention_mask == 0)
    for tok in special_tokens:
        special_mask = special_mask | (input_ids == tok)
    prob_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[replace_mask] = mask_token_id

    random_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_mask
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[random_mask] = random_words[random_mask]
    return input_ids, labels


# ---------- collators ----------

class MLMCollator:
    """MLM masking over PRE-TOKENIZED examples (canonical path). Drop-in replacement
    for HF's DataCollatorForLanguageModeling so we don't depend on a tokenizer object.
    """

    def __init__(self, mask_token_id: int, vocab_size: int, mlm_probability: float = 0.15,
                 pad_token_id: int = 0, special_tokens: Optional[List[int]] = None,
                 max_length: Optional[int] = None):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.special_tokens = set(special_tokens or [])
        self.max_length = max_length  # truncate long sequences to bound padding memory

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        cap = self.max_length or 10**9
        lengths = [min(len(ex["input_ids"]), cap) for ex in examples]
        max_len = max(lengths)

        input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        for i, ex in enumerate(examples):
            L = lengths[i]
            input_ids[i, :L] = torch.as_tensor(ex["input_ids"][:L], dtype=torch.long)
            attention_mask[i, :L] = torch.as_tensor(ex["attention_mask"][:L], dtype=torch.long)

        input_ids, labels = mlm_mask_tokens(
            input_ids, attention_mask, mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size, mlm_probability=self.mlm_probability,
            special_tokens=self.special_tokens,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class RawSmilesMLMCollator:
    """MLM masking over RAW SMILES strings (enumerated path). Optionally randomizes
    each SMILES on the fly (fresh enumeration per visit), tokenizes with the real
    tokenizer, then applies the same 80/10/10 masking as MLMCollator.
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.3, randomize: bool = True,
                 max_length: int = 256, special_tokens: Optional[List[int]] = None):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.randomize = randomize
        self.max_length = max_length
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        if special_tokens is None:
            special_tokens = [t for t in (
                tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.bos_token_id,
                tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id,
            ) if t is not None]
        self.special_tokens = set(special_tokens)
        self._rng = random.Random()

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        from smiles_augment import randomize_smiles
        smiles = [ex["smiles"] for ex in examples]
        if self.randomize:
            smiles = [randomize_smiles(s, self._rng) for s in smiles]
        enc = self.tokenizer(
            smiles, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        input_ids, labels = mlm_mask_tokens(
            enc["input_ids"], enc["attention_mask"], mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size, mlm_probability=self.mlm_probability,
            special_tokens=self.special_tokens,
        )
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}


class MTRCollator:
    """Dense descriptor-MTR (ChemBERTa-2). Raw SMILES → tokens + ~200 normalized
    RDKit descriptor targets computed on the fly (NaN where a descriptor failed →
    masked in the loss)."""

    def __init__(self, tokenizer, descriptor_stats, max_length: int = 128,
                 randomize: bool = False):
        self.tokenizer = tokenizer
        self.stats = descriptor_stats
        self.max_length = max_length
        self.randomize = randomize

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        smiles = [ex["smiles"] for ex in examples]
        if self.randomize:
            from smiles_augment import randomize_smiles
            smiles = [randomize_smiles(s) for s in smiles]
        enc = self.tokenizer(smiles, truncation=True, max_length=self.max_length,
                             padding="longest", return_tensors="pt")
        # descriptors are structure-invariant, so precomputed (already-normalized) targets
        # are valid even under SMILES randomization — skip the RDKit call when present.
        if "descriptors" in examples[0]:
            targets = torch.as_tensor(np.stack([ex["descriptors"] for ex in examples]), dtype=torch.float32)
        else:
            from descriptors_v2 import rdkit_descriptors, normalize
            targets = torch.as_tensor(normalize(rdkit_descriptors(smiles), self.stats), dtype=torch.float32)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "mtr_targets": targets}


# ---------- raw-SMILES streaming (enumerated augmentation path) ----------

class StreamingRawSmilesDataset(StreamingTokenizedDataset):
    """Streams raw SMILES strings from parquet shards with a ``smiles`` column (or a
    pickle list of strings). Inherits worker-sharding + subset_fraction + prefetch
    from StreamingTokenizedDataset; only the per-file reader differs. Yields
    ``{"smiles": <str>}`` for RawSmilesMLMCollator to randomize/tokenize.
    """

    SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "SMILES_canonical", "canonical_smiles", "smiles_canon")

    def _load_file(self, path):
        if path.endswith(".parquet"):
            import pyarrow.dataset as pads
            dataset_path = materialize_path(path) if self.cache_remote_files and is_s3_uri(path) else path
            dataset = parquet_dataset(dataset_path) if is_s3_uri(dataset_path) else pads.dataset(dataset_path, format="parquet")
            cols = dataset.schema.names
            smi_col = next((c for c in self.SMILES_COLUMN_CANDIDATES if c in cols), None)
            if smi_col is None:
                raise ValueError(f"No SMILES column {self.SMILES_COLUMN_CANDIDATES} in {path}; has {cols}")
            # optional precomputed descriptors: companion npy row-aligned to this shard
            desc = None
            desc_dir = getattr(self, "descriptors_dir", None)
            if desc_dir:
                stem = path.rstrip("/").split("/")[-1].replace(".parquet", "")
                dpath = f"{desc_dir.rstrip('/')}/descriptors_{stem}.npy"
                local_d = materialize_path(dpath) if is_s3_uri(dpath) else dpath
                desc = np.load(local_d, mmap_mode="r")  # [n_rows, n_desc], row-aligned
            for batch_idx, batch in enumerate(dataset.to_batches(columns=[smi_col], batch_size=10_000)):
                smis = batch.column(0).to_pylist()
                for i, smi in enumerate(smis):
                    if smi is None:
                        continue
                    key = f"{path}:{batch_idx}:{i}"
                    if not _stable_fraction_membership(key, self.subset_fraction, self.subset_seed):
                        continue
                    if desc is not None:
                        yield {"smiles": smi, "descriptors": np.asarray(desc[batch_idx * 10_000 + i])}
                    else:
                        yield {"smiles": smi}
            return

        # pickle fallback: a list of SMILES strings (or dicts with a smiles key)
        local_path = materialize_path(path) if self.cache_remote_files and is_s3_uri(path) else path
        import pickle
        with open(local_path, "rb") as f:
            obj = pickle.load(f)
        for idx, row in enumerate(obj):
            smi = row["smiles"] if isinstance(row, dict) else row
            key = f"{path}:{idx}"
            if not _stable_fraction_membership(key, self.subset_fraction, self.subset_seed):
                continue
            yield {"smiles": smi}


def make_raw_smiles_dataset(
    paths: List[str],
    subset_fraction: Optional[float] = None,
    subset_seed: int = 0,
    cache_remote_files: bool = True,
    descriptors_dir: Optional[str] = None,
) -> StreamingRawSmilesDataset:
    ds = StreamingRawSmilesDataset(
        paths, with_labels=False, shuffle=True,
        subset_fraction=subset_fraction, subset_seed=subset_seed,
        cache_remote_files=cache_remote_files,
    )
    ds.descriptors_dir = descriptors_dir  # companion precomputed descriptors (MTR speedup)
    return ds


class SupervisedCollator:
    """Pads input_ids and attention_mask; passes labels through."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(ex["input_ids"].shape[0] for ex in examples)
        input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        labels = torch.stack([ex["labels"] for ex in examples], dim=0)
        for i, ex in enumerate(examples):
            L = ex["input_ids"].shape[0]
            input_ids[i, :L] = ex["input_ids"]
            attention_mask[i, :L] = ex["attention_mask"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------- mixed-mode batch iterator ----------

class MixedModeBatchIterator:
    """Yields (mode, batch) pairs. Mode is 'mlm' or 'sup'. Per-batch, sample mode
    independently with P(mlm) = mixing_ratio. One optimizer step per batch.

    The iterator ends when total_batches batches have been yielded.
    """

    def __init__(
        self,
        mlm_loader: Optional[DataLoader],
        sup_loader: Optional[DataLoader],
        mixing_ratio: float,
        total_batches: int,
        seed: int = 0,
    ):
        if mixing_ratio < 1.0 and sup_loader is None:
            raise ValueError("mixing_ratio < 1 requires a supervised loader")
        if mixing_ratio > 0.0 and mlm_loader is None:
            raise ValueError("mixing_ratio > 0 requires an MLM loader")
        self.mlm_loader = mlm_loader
        self.sup_loader = sup_loader
        self.mixing_ratio = mixing_ratio
        self.total_batches = total_batches
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
        mlm_iter = iter(self.mlm_loader) if self.mlm_loader is not None else None
        sup_iter = iter(self.sup_loader) if self.sup_loader is not None else None

        def restart_mlm():
            return iter(self.mlm_loader)

        def restart_sup():
            return iter(self.sup_loader)

        for step in range(self.total_batches):
            choose_mlm = self.rng.random() < self.mixing_ratio
            if choose_mlm and mlm_iter is not None:
                try:
                    batch = next(mlm_iter)
                except StopIteration:
                    mlm_iter = restart_mlm()
                    batch = next(mlm_iter)
                yield "mlm", batch
            else:
                try:
                    batch = next(sup_iter)
                except StopIteration:
                    sup_iter = restart_sup()
                    batch = next(sup_iter)
                yield "sup", batch


class MultiObjectiveBatchIterator:
    """Generalizes MixedModeBatchIterator to N objectives (e.g. mlm/mtr/supervised).
    Each batch samples an objective ~ weights, draws from that objective's loader
    (restarting on exhaustion), and yields (objective, batch)."""

    def __init__(self, loaders: Dict[str, DataLoader], weights: Dict[str, float],
                 total_batches: int, seed: int = 0):
        self.loaders = {k: v for k, v in loaders.items() if v is not None and weights.get(k, 0) > 0}
        self.objs = list(self.loaders.keys())
        self.weights = [float(weights[k]) for k in self.objs]
        if not self.objs:
            raise ValueError("MultiObjectiveBatchIterator needs at least one weighted loader")
        self.total_batches = total_batches
        self.rng = random.Random(seed)

    def __iter__(self):
        iters = {k: iter(v) for k, v in self.loaders.items()}
        for _ in range(self.total_batches):
            obj = self.rng.choices(self.objs, weights=self.weights, k=1)[0]
            try:
                batch = next(iters[obj])
            except StopIteration:
                iters[obj] = iter(self.loaders[obj])
                batch = next(iters[obj])
            yield obj, batch

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

from data import StreamingTokenizedDataset
from storage_utils import is_s3_uri, materialize_path, parquet_dataset


# ---------- supervised in-RAM loader ----------

def _family_columns(schema_columns: List[str], families: List[str]) -> List[str]:
    """Pick label columns belonging to the given families. Convention: column names
    start with `<family>__`. We exclude `input_ids`, `attention_mask`, and any column
    that doesn't match the convention.
    """
    out = []
    for col in schema_columns:
        if col in ("input_ids", "attention_mask", "smiles", "canonical_smiles"):
            continue
        for fam in families:
            if col.startswith(f"{fam}__") or col == fam:
                out.append(col)
                break
    return out


def load_supervised_inram(
    parquet_path: str,
    families: List[str],
    max_length: int = 512,
    max_rows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Load (input_ids, attention_mask, labels) into RAM for the requested families.

    Returns:
        input_ids: int32 tensor [N, max_length]
        attention_mask: int8 tensor [N, max_length]
        labels: float32 tensor [N, T] where T = #label columns; NaN for missing
        column_names: ordered list of label columns
        task_types: per-column "classification" or "regression" inferred from values
    """
    import pyarrow.dataset as pads

    if is_s3_uri(parquet_path):
        ds_obj = parquet_dataset(parquet_path)
    else:
        ds_obj = pads.dataset(parquet_path, format="parquet")

    schema_cols = ds_obj.schema.names
    label_cols = _family_columns(schema_cols, families)
    if not label_cols:
        raise ValueError(
            f"No label columns matched families {families} in parquet {parquet_path}"
        )
    if "input_ids" not in schema_cols or "attention_mask" not in schema_cols:
        raise ValueError(f"Parquet {parquet_path} missing input_ids/attention_mask columns")

    columns = ["input_ids", "attention_mask"] + label_cols

    ids_chunks, mask_chunks, label_chunks = [], [], []
    rows_seen = 0
    print(f"[load_supervised_inram] streaming {len(label_cols)} label cols across families {families}", flush=True)
    for batch_idx, batch in enumerate(ds_obj.to_batches(columns=columns, batch_size=8192)):
        ids_arrow = batch.column(batch.schema.get_field_index("input_ids"))
        mask_arrow = batch.column(batch.schema.get_field_index("attention_mask"))
        n = len(ids_arrow)

        # Vectorised pad: flatten, then place via offsets.
        ids_flat = np.asarray(ids_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int32)
        mask_flat = np.asarray(mask_arrow.values.to_numpy(zero_copy_only=False), dtype=np.int8)
        ids_offsets = ids_arrow.offsets.to_numpy(zero_copy_only=False)
        mask_offsets = mask_arrow.offsets.to_numpy(zero_copy_only=False)

        ids_arr = np.zeros((n, max_length), dtype=np.int32)
        mask_arr = np.zeros((n, max_length), dtype=np.int8)
        for i in range(n):
            s_id, e_id = int(ids_offsets[i]), int(ids_offsets[i + 1])
            s_m, e_m = int(mask_offsets[i]), int(mask_offsets[i + 1])
            L_id = min(e_id - s_id, max_length)
            L_m = min(e_m - s_m, max_length)
            if L_id > 0:
                ids_arr[i, :L_id] = ids_flat[s_id : s_id + L_id]
            if L_m > 0:
                mask_arr[i, :L_m] = mask_flat[s_m : s_m + L_m]
        ids_chunks.append(ids_arr)
        mask_chunks.append(mask_arr)

        # Vectorised labels: each col → numpy with NaN for nulls.
        labels_arr = np.full((n, len(label_cols)), np.nan, dtype=np.float32)
        for j, col in enumerate(label_cols):
            arr = batch.column(batch.schema.get_field_index(col))
            # to_numpy returns object dtype if nulls present; convert.
            np_col = arr.to_numpy(zero_copy_only=False)
            if np_col.dtype == np.object_:
                # has nulls — convert with NaN replacement
                col_floats = np.array([float(v) if v is not None else np.nan for v in np_col], dtype=np.float32)
            else:
                col_floats = np_col.astype(np.float32, copy=False)
            labels_arr[:, j] = col_floats
        label_chunks.append(labels_arr)

        rows_seen += n
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}: {rows_seen} rows seen", flush=True)
        if max_rows is not None and rows_seen >= max_rows:
            break

    input_ids = torch.from_numpy(np.concatenate(ids_chunks, axis=0)).to(torch.int64)
    attention_mask = torch.from_numpy(np.concatenate(mask_chunks, axis=0)).to(torch.int64)
    labels = torch.from_numpy(np.concatenate(label_chunks, axis=0))

    if max_rows is not None:
        input_ids = input_ids[:max_rows]
        attention_mask = attention_mask[:max_rows]
        labels = labels[:max_rows]

    # Drop rows where every label is NaN (no signal for any active family)
    has_label = (~torch.isnan(labels)).any(dim=1)
    input_ids = input_ids[has_label]
    attention_mask = attention_mask[has_label]
    labels = labels[has_label]

    # Infer task type per column: if all non-NaN values are in {0,1}, classification.
    task_types = []
    for j in range(labels.shape[1]):
        col = labels[:, j]
        col = col[~torch.isnan(col)]
        if col.numel() == 0:
            task_types.append("regression")
            continue
        unique_vals = torch.unique(col)
        if unique_vals.numel() <= 2 and torch.all((unique_vals == 0) | (unique_vals == 1)):
            task_types.append("classification")
        else:
            task_types.append("regression")

    return input_ids, attention_mask, labels, label_cols, task_types


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


# ---------- collators ----------

class MLMCollator:
    """MLM masking. Drop-in replacement for HF's DataCollatorForLanguageModeling
    so we don't depend on a tokenizer object at training time.
    """

    def __init__(self, mask_token_id: int, vocab_size: int, mlm_probability: float = 0.15,
                 pad_token_id: int = 0, special_tokens: Optional[List[int]] = None):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.special_tokens = set(special_tokens or [])

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)

        input_ids = torch.full((len(examples), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        for i, ex in enumerate(examples):
            ids = torch.as_tensor(ex["input_ids"], dtype=torch.long)
            mask = torch.as_tensor(ex["attention_mask"], dtype=torch.long)
            L = ids.shape[0]
            input_ids[i, :L] = ids
            attention_mask[i, :L] = mask

        labels = input_ids.clone()
        prob_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask padding or special tokens
        special_mask = (attention_mask == 0)
        for tok in self.special_tokens:
            special_mask = special_mask | (input_ids == tok)
        prob_matrix.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80% replace with [MASK]
        replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[replace_mask] = self.mask_token_id

        # 10% random token (of remaining 20%)
        random_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_mask
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[random_mask] = random_words[random_mask]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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

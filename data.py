from __future__ import annotations

import hashlib
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from storage_utils import is_s3_uri, list_data_files, materialize_path, parquet_dataset


try:
    import pyarrow.dataset as ds
except Exception as exc:
    ds = None
    _pyarrow_import_error = exc
else:
    _pyarrow_import_error = None


def _expand_paths(path):
    return list_data_files(path)


def _stable_fraction_membership(key: str, fraction: Optional[float], seed: int) -> bool:
    if fraction is None or fraction >= 1.0:
        return True
    if fraction <= 0:
        return False
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    return value < fraction


class StreamingTokenizedDataset(torch.utils.data.IterableDataset):
    """Iterable dataset for tokenized pickles, suitable for very large data."""

    def __init__(
        self,
        paths,
        with_labels: bool = False,
        shuffle: bool = False,
        max_samples: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        subset_seed: int = 0,
        cache_remote_files: bool = True,
        prefetch_files: int = 2,
    ):
        self.paths = []
        for path in paths:
            self.paths.extend(_expand_paths(path))
        if not self.paths:
            raise ValueError("No tokenized files provided for streaming dataset")
        self.with_labels = with_labels
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.cache_remote_files = cache_remote_files
        self.prefetch_files = max(prefetch_files, 0)

    def _load_file(self, path):
        if path.endswith(".parquet"):
            if ds is None:
                raise ImportError(
                    "pyarrow is required to stream parquet tokenized data. "
                    f"Import error: {_pyarrow_import_error}"
                )
            dataset_path = materialize_path(path) if self.cache_remote_files and is_s3_uri(path) else path
            dataset = parquet_dataset(dataset_path) if is_s3_uri(dataset_path) else ds.dataset(dataset_path, format="parquet")
            columns = dataset.schema.names
            if "input_ids" not in columns or "attention_mask" not in columns:
                raise ValueError(f"Parquet missing input_ids/attention_mask: {path}")
            want_labels = self.with_labels
            for batch_idx, batch in enumerate(dataset.to_batches(columns=columns, batch_size=10_000)):
                ids_all = batch.column(batch.schema.get_field_index("input_ids")).to_pylist()
                mask_all = batch.column(batch.schema.get_field_index("attention_mask")).to_pylist()
                labels_all = None
                if want_labels:
                    if "labels" not in columns:
                        raise ValueError("Requested labels but no labels column in parquet")
                    labels_all = batch.column(batch.schema.get_field_index("labels")).to_pylist()

                for i, ids in enumerate(ids_all):
                    sample_key = f"{path}:{batch_idx}:{i}"
                    if not _stable_fraction_membership(sample_key, self.subset_fraction, self.subset_seed):
                        continue
                    item = {"input_ids": ids, "attention_mask": mask_all[i]}
                    if want_labels:
                        item["labels"] = torch.tensor(labels_all[i], dtype=torch.float32)
                    yield item
            return

        local_path = materialize_path(path) if self.cache_remote_files and is_s3_uri(path) else path
        with open(local_path, 'rb') as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            tokenized = obj.get('data', obj)
            labels = obj.get('labels')
        else:
            tokenized = obj
            labels = None

        if self.with_labels and labels is None:
            raise ValueError("Requested labels but none found in pickle")

        for idx, sample in enumerate(tokenized):
            sample_key = f"{path}:{idx}"
            if not _stable_fraction_membership(sample_key, self.subset_fraction, self.subset_seed):
                continue
            item = sample.copy()
            if self.with_labels:
                item['labels'] = torch.tensor(labels[idx], dtype=torch.float32)
            yield item

    def _iter_paths(self, paths):
        if self.shuffle:
            rng = random.Random(self.subset_seed)
            rng.shuffle(paths)
        total = 0
        with ThreadPoolExecutor(max_workers=max(self.prefetch_files, 1)) as executor:
            futures = {}

            def _schedule(target_path):
                if not self.cache_remote_files or not is_s3_uri(target_path):
                    return
                if target_path in futures:
                    return
                futures[target_path] = executor.submit(materialize_path, target_path)

            for idx, path in enumerate(paths):
                for offset in range(1, self.prefetch_files + 1):
                    nxt = idx + offset
                    if nxt < len(paths):
                        _schedule(paths[nxt])
                for item in self._load_file(path):
                    yield item
                    total += 1
                    if self.max_samples and total >= self.max_samples:
                        return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker_info:
            paths = paths[worker_info.id::worker_info.num_workers]
        return self._iter_paths(paths)


def _family_columns(schema, families: List[str]) -> List[str]:

    import pyarrow as pa
    is_schema = hasattr(schema, "field")
    schema_columns = schema.names if is_schema else list(schema)

    EXCLUDE = {"input_ids", "attention_mask", "smiles", "SMILES", "canonical_smiles", "canonical_SMILES"}

    def _is_numeric(name: str) -> bool:
        if not is_schema:
            return True
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

    import pyarrow.dataset as pads

    ds_obj = parquet_dataset(parquet_path) if is_s3_uri(parquet_path) else pads.dataset(parquet_path, format="parquet")
    schema = ds_obj.schema
    if "input_ids" not in schema.names or "attention_mask" not in schema.names:
        raise ValueError(f"Parquet {parquet_path} missing input_ids/attention_mask")


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


    route_order = sorted(families, key=lambda f: int(family_caps.get(f, 100_000)))
    n_blocked = 0
    print(f"[stratified] families={families} route_order={route_order} "
          f"caps={ {f:family_caps.get(f) for f in families} } cols={len(label_cols)} "
          f"blocklist={len(blocklist) if blocklist else 0}", flush=True)

    for batch in ds_obj.to_batches(columns=columns, batch_size=8192):
        if all(kept[f] >= int(family_caps.get(f, 0)) for f in families):
            break
        n = batch.num_rows

        bl = np.full((n, len(label_cols)), np.nan, dtype=np.float64)
        for j, c in enumerate(label_cols):
            arr = batch.column(batch.schema.get_field_index(c)).to_numpy(zero_copy_only=False)
            bl[:, j] = np.array([float(v) if v is not None else np.nan for v in arr], dtype=np.float64) \
                if arr.dtype == np.object_ else arr.astype(np.float64, copy=False)

        blocked = np.zeros(n, dtype=bool)
        if blocklist and have_smiles:
            sc = batch.column(batch.schema.get_field_index("smiles_canon")).to_pylist()
            blocked = np.array([(s in blocklist) for s in sc], dtype=bool)
            n_blocked += int(blocked.sum())

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


def make_mlm_dataset(
    paths: List[str],
    max_samples: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    subset_seed: int = 0,
    cache_remote_files: bool = True,
) -> StreamingTokenizedDataset:

    return StreamingTokenizedDataset(
        paths,
        with_labels=False,
        shuffle=True,
        max_samples=max_samples,
        subset_fraction=subset_fraction,
        subset_seed=subset_seed,
        cache_remote_files=cache_remote_files,
    )


def mlm_mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    special_tokens: set,
) -> Tuple[torch.Tensor, torch.Tensor]:

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


class MLMCollator:


    def __init__(self, mask_token_id: int, vocab_size: int, mlm_probability: float = 0.15,
                 pad_token_id: int = 0, special_tokens: Optional[List[int]] = None,
                 max_length: Optional[int] = None):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.pad_token_id = pad_token_id
        self.special_tokens = set(special_tokens or [])
        self.max_length = max_length

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


        if "descriptors" in examples[0]:
            targets = torch.as_tensor(np.stack([ex["descriptors"] for ex in examples]), dtype=torch.float32)
        else:
            from descriptors import rdkit_descriptors, normalize
            targets = torch.as_tensor(normalize(rdkit_descriptors(smiles), self.stats), dtype=torch.float32)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "mtr_targets": targets}


class CorruptedCollator:


    def __init__(self, base, mode: str, seed: int = 0):
        valid = {"shuffle_tokens", "shuffle_targets", "both"}
        if mode not in valid:
            raise ValueError(f"corruption mode must be one of {sorted(valid)}, got {mode!r}")
        self.base = base
        self.mode = mode
        self.seed = int(seed)
        self._g = None

    def _gen(self) -> torch.Generator:
        if self._g is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._g = torch.Generator().manual_seed(self.seed * 100003 + wid)
        return self._g

    def _shuffle_tokens(self, batch: Dict[str, torch.Tensor]) -> None:
        ids, am = batch["input_ids"], batch["attention_mask"]
        labels = batch.get("labels")
        g = self._gen()
        for i in range(ids.size(0)):
            L = int(am[i].sum().item())
            if L <= 3:
                continue
            idx = torch.arange(1, L - 1)
            perm = idx[torch.randperm(idx.numel(), generator=g)]
            ids[i, idx] = ids[i, perm].clone()
            if labels is not None:
                labels[i, idx] = labels[i, perm].clone()

    def _shuffle_targets(self, batch: Dict[str, torch.Tensor]) -> None:
        t = batch.get("mtr_targets")
        if t is None or t.size(0) < 2:
            return
        g = self._gen()
        n = t.size(0)
        perm = torch.randperm(n, generator=g)
        if bool((perm == torch.arange(n)).all()):
            perm = torch.roll(perm, 1)
        batch["mtr_targets"] = t[perm].clone()

    def __call__(self, examples):
        batch = self.base(examples)
        if self.mode in ("shuffle_tokens", "both") and "input_ids" in batch:
            self._shuffle_tokens(batch)
        if self.mode in ("shuffle_targets", "both") and "mtr_targets" in batch:
            self._shuffle_targets(batch)
        return batch


class StreamingRawSmilesDataset(StreamingTokenizedDataset):


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

            desc = None
            desc_dir = getattr(self, "descriptors_dir", None)
            if desc_dir:
                stem = path.rstrip("/").split("/")[-1].replace(".parquet", "")
                dpath = f"{desc_dir.rstrip('/')}/descriptors_{stem}.npy"
                local_d = materialize_path(dpath) if is_s3_uri(dpath) else dpath
                desc = np.load(local_d, mmap_mode="r")
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
    ds.descriptors_dir = descriptors_dir
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


class MultiObjectiveBatchIterator:


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

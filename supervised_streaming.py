"""supervised_streaming.py
Streaming supervised dataset utilities for wide parquet files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from tasks import TaskRegistry, TaskSpec, TaskType

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - handled at runtime
    pa = None
    pc = None
    ds = None
    pq = None
    _pyarrow_import_error = exc
else:
    _pyarrow_import_error = None


DEFAULT_FAMILIES = [
    {"name": "PCQM", "prefix": "PCQM__"},
    {"name": "WONG", "prefix": "WONG__"},
    {"name": "L1000_MCF7", "prefix": "L1000_MCF7__"},
    {"name": "L1000_VCAP", "prefix": "L1000_VCAP__"},
    {"name": "PCBA", "prefix": "PCBA__"},
]


def _require_pyarrow():
    if pa is None:
        raise ImportError(
            "pyarrow is required for supervised parquet streaming. "
            f"Import error: {_pyarrow_import_error}"
        )


def detect_smiles_column(columns: List[str]) -> str:
    candidates = ["smiles_canon", "SMILES_std", "SMILES", "smiles"]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"No SMILES column found. Tried: {candidates}")


def resolve_family_specs(
    parquet_path: str,
    families: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, object]]]:
    _require_pyarrow()
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)

    schema = pq.ParquetFile(path).schema
    columns = schema.names
    smiles_col = detect_smiles_column(columns)

    family_cfgs = families or DEFAULT_FAMILIES
    resolved = []
    for fam in family_cfgs:
        name = fam["name"]
        prefix = fam.get("prefix", f"{name}__")
        if not prefix.endswith("__"):
            prefix = f"{prefix}__"
        cols = [c for c in columns if c.startswith(prefix)]
        resolved.append(
            {
                "name": name,
                "prefix": prefix,
                "columns": cols,
            }
        )
    return smiles_col, resolved


def _is_float_array(arr: pa.Array) -> bool:
    return pa.types.is_floating(arr.type)


def count_non_nan_labels(parquet_path: str, columns: List[str], batch_rows: int = 50_000) -> int:
    _require_pyarrow()
    if not columns:
        return 0
    dataset = ds.dataset(parquet_path, format="parquet")
    total = 0
    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        for col in columns:
            arr = batch.column(col)
            valid = pc.invert(pc.is_null(arr))
            if _is_float_array(arr):
                valid = pc.and_(valid, pc.invert(pc.is_nan(arr)))
            total += int(pc.sum(valid).as_py() or 0)
    return total


def estimate_avg_tokens_from_parquet(
    parquet_path: str,
    smiles_col: str,
    tokenizer,
    sample_size: int = 2000,
    max_length: int = 512,
) -> float:
    _require_pyarrow()
    pf = pq.ParquetFile(parquet_path)
    seen = 0
    lengths: List[int] = []
    for batch in pf.iter_batches(columns=[smiles_col], batch_size=10_000):
        smiles_all = batch.column(smiles_col).to_pylist()
        smiles = [s for s in smiles_all if s]
        if not smiles:
            continue
        if seen + len(smiles) > sample_size:
            smiles = smiles[: max(sample_size - seen, 0)]
        encoded = tokenizer(
            smiles,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        lengths.extend(len(x) for x in input_ids)
        seen += len(smiles)
        if seen >= sample_size:
            break
    if not lengths:
        return float(max_length)
    return float(np.mean(lengths))


def build_task_registry_for_family(columns: List[str]) -> TaskRegistry:
    registry = TaskRegistry()
    for col in columns:
        registry.register(TaskSpec(name=col, task_type=TaskType.REGRESSION))
    return registry


@dataclass
class SupervisedFamily:
    name: str
    prefix: str
    columns: List[str]
    label_count: int = 0


class StreamingSupervisedFamilyDataset(IterableDataset):
    """Stream a supervised family from a wide parquet file."""

    def __init__(
        self,
        parquet_path: str,
        smiles_col: str,
        label_columns: List[str],
        tokenizer,
        max_length: int = 512,
        batch_rows: int = 4096,
    ) -> None:
        _require_pyarrow()
        if not label_columns:
            raise ValueError("No label columns provided for supervised streaming dataset")
        self.parquet_path = parquet_path
        self.smiles_col = smiles_col
        self.label_columns = label_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_rows = batch_rows

    def __iter__(self) -> Iterable[Dict[str, object]]:
        dataset = ds.dataset(self.parquet_path, format="parquet")
        columns = [self.smiles_col] + self.label_columns
        for batch in dataset.to_batches(columns=columns, batch_size=self.batch_rows):
            smiles_all = batch.column(self.smiles_col).to_pylist()
            valid_idx = [i for i, s in enumerate(smiles_all) if s]
            if not valid_idx:
                continue
            smiles = [smiles_all[i] for i in valid_idx]
            encoded = self.tokenizer(
                smiles,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            n = encoded["input_ids"].shape[0]
            values: Dict[str, np.ndarray] = {}
            masks: Dict[str, np.ndarray] = {}
            for col in self.label_columns:
                arr = batch.column(col).to_numpy(zero_copy_only=False)
                vals = np.asarray(arr, dtype=np.float32)[valid_idx]
                mask = ~np.isnan(vals)
                values[col] = vals
                masks[col] = mask

            for i in range(n):
                labels = {col: torch.tensor(values[col][i], dtype=torch.float32) for col in self.label_columns}
                label_masks = {col: torch.tensor(masks[col][i], dtype=torch.float32) for col in self.label_columns}
                yield {
                    "input_ids": encoded["input_ids"][i],
                    "attention_mask": encoded["attention_mask"][i],
                    "labels": labels,
                    "label_masks": label_masks,
                }

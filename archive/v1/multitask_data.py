# ==============================================================================
# FILE: multitask_data.py
# Multi-task dataset with SMILES-level pooling for large-scale pretraining
# ==============================================================================

"""
Multi-task data loading for large-scale chemical pretraining.

Key Concepts:
- Each unique SMILES appears ONCE in the final dataset
- Labels are pooled from multiple data sources
- Missing labels are masked (not included in loss)
- Supervised losses computed in model (this module provides data)

User Input Flow:
1. Define tasks in TaskRegistry (name, type, metric)
2. Specify data sources with label mappings:
   {
       'path': 'data/bbbp.csv',
       'SMILES_column': 'SMILES',
       'label_mapping': {'p_np': 'BBBP'}  # csv_column -> task_name
   }
3. Pool all sources into one dataset by unique SMILES
4. Train with masked losses (MultiTaskModel handles loss computation)

Example Config:
    tasks:
      - name: BBBP
        type: binary_classification
      - name: ESOL
        type: regression

    data_sources:
      - path: data/bbbp.csv
        SMILES_column: SMILES
        label_mapping:
          p_np: BBBP
      - path: data/esol.csv
        SMILES_column: SMILES
        label_mapping:
          measured_log_solubility: ESOL
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tasks import TaskRegistry, TaskType

logger = logging.getLogger(__name__)


@dataclass
class DataSourceSpec:
    """Specification for a data source."""
    path: str
    SMILES_column: str = "SMILES"
    label_mapping: Optional[Dict[str, str]] = None  # csv_column -> task_name

    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {}


class MultiTaskDataset(Dataset):
    """
    Pooled multi-task dataset with unique SMILES.

    Provides:
    - input_ids, attention_mask: tokenized SMILES
    - labels: Dict[task_name -> tensor]
    - label_masks: Dict[task_name -> tensor] (1=valid, 0=missing)
    """

    def __init__(
        self,
        tokenized_data: List[Dict[str, Any]],
        labels: Dict[str, np.ndarray],
        task_registry: TaskRegistry,
        SMILES_list: Optional[List[str]] = None,
    ):
        """
        Args:
            tokenized_data: Pre-tokenized samples with input_ids, attention_mask
            labels: Dict[task_name -> np.array] with np.nan for missing
            task_registry: Registry defining all tasks
            SMILES_list: Original SMILES strings (optional, for debugging)
        """
        self.data = tokenized_data
        self.task_registry = task_registry
        self.task_names = task_registry.task_names
        self.SMILES_list = SMILES_list

        n = len(tokenized_data)
        self._n_samples = n

        # Build label and mask arrays for ALL registered tasks
        self.labels = {}
        self.masks = {}

        for task_name in self.task_names:
            if task_name in labels:
                arr = np.array(labels[task_name], dtype=np.float32)
                assert len(arr) == n, f"Task {task_name}: {len(arr)} labels != {n} samples"
            else:
                arr = np.full(n, np.nan, dtype=np.float32)

            mask = ~np.isnan(arr)
            self.labels[task_name] = np.nan_to_num(arr, nan=0.0)
            self.masks[task_name] = mask.astype(np.float32)

        self._log_stats()

    def _log_stats(self):
        """Log dataset statistics."""
        logger.info(f"MultiTaskDataset: {self._n_samples} samples, {len(self.task_names)} tasks")
        total_labels = 0
        for task_name in self.task_names:
            n_valid = int(self.masks[task_name].sum())
            total_labels += n_valid
            if n_valid > 0:
                pct = 100 * n_valid / self._n_samples
                logger.info(f"  {task_name}: {n_valid} labels ({pct:.1f}%)")
        avg_labels = total_labels / self._n_samples if self._n_samples > 0 else 0
        logger.info(f"  Average labels per sample: {avg_labels:.2f}")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'labels': {},
            'label_masks': {},
        }

        for task_name in self.task_names:
            task_spec = self.task_registry.get(task_name)
            val = self.labels[task_name][idx]
            mask = self.masks[task_name][idx]

            # Type depends on task
            if task_spec.task_type == TaskType.REGRESSION:
                item['labels'][task_name] = torch.tensor(val, dtype=torch.float32)
            else:
                item['labels'][task_name] = torch.tensor(int(val), dtype=torch.long)

            item['label_masks'][task_name] = torch.tensor(mask, dtype=torch.float32)

        return item

    def get_coverage(self) -> Dict[str, float]:
        """Get label coverage (fraction with valid labels) per task."""
        return {
            task: self.masks[task].sum() / self._n_samples
            for task in self.task_names
        }

    def save(self, path: str) -> None:
        """Save dataset to pickle."""
        # Store labels with NaN restored
        labels_with_nan = {}
        for task in self.task_names:
            arr = self.labels[task].copy()
            arr[self.masks[task] == 0] = np.nan
            labels_with_nan[task] = arr

        obj = {
            'data': self.data,
            'labels': labels_with_nan,
            'task_names': self.task_names,
            'SMILES_list': self.SMILES_list,
        }
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved to {path}")

    @classmethod
    def load(cls, path: str, task_registry: TaskRegistry) -> 'MultiTaskDataset':
        """Load dataset from pickle."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Loaded {len(obj['data'])} samples from {path}")
        return cls(
            tokenized_data=obj['data'],
            labels=obj['labels'],
            task_registry=task_registry,
            SMILES_list=obj.get('SMILES_list'),
        )


class MultiTaskCollator:
    """Collate multi-task batches."""

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        result = {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        }

        if 'labels' in batch[0]:
            task_names = list(batch[0]['labels'].keys())
            result['labels'] = {
                t: torch.stack([b['labels'][t] for b in batch])
                for t in task_names
            }
            result['label_masks'] = {
                t: torch.stack([b['label_masks'][t] for b in batch])
                for t in task_names
            }

        return result


# ==============================================================================
# Data Loading and Pooling
# ==============================================================================

def load_data_source(
    spec: DataSourceSpec,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Load a single data source.

    Returns:
        (SMILES_list, labels_dict) where labels_dict maps task_name -> values
    """
    path = spec.path

    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.pkl') or path.endswith('.pickle'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    SMILES_list = df[spec.SMILES_column].tolist()

    labels = {}
    for col_name, task_name in spec.label_mapping.items():
        if col_name in df.columns:
            labels[task_name] = df[col_name].values.astype(np.float32)
        else:
            logger.warning(f"Column '{col_name}' not in {path}, skipping")

    logger.info(f"Loaded {len(SMILES_list)} samples from {path}")
    for task, vals in labels.items():
        n_valid = np.sum(~np.isnan(vals))
        logger.info(f"  {task}: {n_valid} valid labels")

    return SMILES_list, labels


def pool_by_SMILES(
    data_sources: List[Tuple[List[str], Dict[str, np.ndarray]]],
    task_registry: TaskRegistry,
    tokenizer,
    max_length: int = 512,
) -> MultiTaskDataset:
    """
    Pool multiple data sources by unique SMILES.

    For duplicate SMILES, labels are merged (first non-NaN wins).

    Args:
        data_sources: List of (SMILES_list, labels_dict) tuples
        task_registry: Registry of all tasks
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length for tokenization

    Returns:
        MultiTaskDataset with unique SMILES
    """
    # Aggregate by SMILES
    SMILES_to_labels: Dict[str, Dict[str, float]] = defaultdict(dict)
    SMILES_order = []
    seen = set()

    total_samples = 0
    for SMILES_list, labels in data_sources:
        total_samples += len(SMILES_list)

        for i, smi in enumerate(SMILES_list):
            # Track first occurrence order
            if smi not in seen:
                SMILES_order.append(smi)
                seen.add(smi)

            # Merge labels (first valid value wins)
            for task_name, task_vals in labels.items():
                val = task_vals[i]
                existing = SMILES_to_labels[smi].get(task_name, np.nan)
                if np.isnan(existing) and not np.isnan(val):
                    SMILES_to_labels[smi][task_name] = val

    logger.info(f"Pooled {total_samples} samples -> {len(SMILES_order)} unique SMILES")

    # Tokenize
    logger.info("Tokenizing...")
    tokenized = tokenizer(
        SMILES_order,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    tokenized_data = [
        {
            'input_ids': torch.tensor(tokenized['input_ids'][i]),
            'attention_mask': torch.tensor(tokenized['attention_mask'][i]),
        }
        for i in range(len(SMILES_order))
    ]

    # Build label arrays
    labels = {}
    for task_name in task_registry.task_names:
        arr = np.array([
            SMILES_to_labels[smi].get(task_name, np.nan)
            for smi in SMILES_order
        ], dtype=np.float32)
        labels[task_name] = arr

    return MultiTaskDataset(
        tokenized_data=tokenized_data,
        labels=labels,
        task_registry=task_registry,
        SMILES_list=SMILES_order,
    )


def create_dataset_from_sources(
    source_specs: List[DataSourceSpec],
    task_registry: TaskRegistry,
    tokenizer,
    max_length: int = 512,
) -> MultiTaskDataset:
    """
    Create pooled dataset from multiple data source specifications.

    Args:
        source_specs: List of DataSourceSpec objects
        task_registry: Registry of all tasks
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length

    Returns:
        MultiTaskDataset
    """
    data_sources = []
    for spec in source_specs:
        SMILES_list, labels = load_data_source(spec)
        data_sources.append((SMILES_list, labels))

    return pool_by_SMILES(data_sources, task_registry, tokenizer, max_length)


def create_dataloader(
    dataset: MultiTaskDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader with multi-task collator."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MultiTaskCollator(),
        pin_memory=True,
    )


def train_val_split(
    dataset: MultiTaskDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[MultiTaskDataset, MultiTaskDataset]:
    """
    Random train/val split.

    Note: For production, use scaffold splitting instead.
    """
    n = len(dataset)
    n_val = int(n * val_fraction)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    def make_subset(idx: np.ndarray) -> MultiTaskDataset:
        labels_subset = {
            task: dataset.labels[task][idx]
            for task in dataset.task_names
        }
        # Restore NaN for masked values
        for task in dataset.task_names:
            mask = dataset.masks[task][idx]
            labels_subset[task][mask == 0] = np.nan

        return MultiTaskDataset(
            tokenized_data=[dataset.data[i] for i in idx],
            labels=labels_subset,
            task_registry=dataset.task_registry,
            SMILES_list=[dataset.SMILES_list[i] for i in idx] if dataset.SMILES_list else None,
        )

    return make_subset(train_idx), make_subset(val_idx)

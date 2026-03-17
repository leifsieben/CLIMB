"""data.py"""

import hashlib
import random
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, ConcatDataset, IterableDataset

try:
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - handled at runtime
    ds = None
    _pyarrow_import_error = exc
else:
    _pyarrow_import_error = None

from storage_utils import is_s3_uri, list_data_files, materialize_path, parquet_dataset


class UnsupervisedChemicalDataset(Dataset):
    """Pre-tokenized dataset for MLM"""
    
    def __init__(self, tokenized_data: List[dict]):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def save(tokenized_data: List[dict], path: str):
        """Save pre-tokenized data"""
        with open(path, 'wb') as f:
            pickle.dump(tokenized_data, f)
        print(f"Saved {len(tokenized_data)} samples to {path}")
    
    @staticmethod
    def load(path: str) -> 'UnsupervisedChemicalDataset':
        """Load pre-tokenized data"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} samples from {path}")
        return UnsupervisedChemicalDataset(data)


class SupervisedChemicalDataset(Dataset):
    """Pre-tokenized dataset for supervised learning"""
    
    def __init__(self, tokenized_data: List[dict], labels: List[float]):
        assert len(tokenized_data) == len(labels)
        self.data = tokenized_data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    @staticmethod
    def save(tokenized_data: List[dict], labels: List[float], path: str):
        """Save pre-tokenized data with labels"""
        with open(path, 'wb') as f:
            pickle.dump({'data': tokenized_data, 'labels': labels}, f)
        print(f"Saved {len(tokenized_data)} samples to {path}")
    
    @staticmethod
    def load(path: str) -> 'SupervisedChemicalDataset':
        """Load pre-tokenized data with labels"""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Loaded {len(obj['data'])} samples from {path}")
        return SupervisedChemicalDataset(obj['data'], obj['labels'])


class MixedDataset(ConcatDataset):
    """Mix unsupervised and supervised datasets"""
    
    def __init__(
        self,
        unsupervised_dataset: Optional[UnsupervisedChemicalDataset],
        supervised_dataset: Optional[SupervisedChemicalDataset],
        unsupervised_weight: float = 0.5,
        supervised_weight: float = 0.5,
    ):
        datasets = []
        
        if unsupervised_dataset and unsupervised_weight > 0:
            n_unsup = int(len(unsupervised_dataset) * unsupervised_weight)
            datasets.append(torch.utils.data.Subset(unsupervised_dataset, range(n_unsup)))
        
        if supervised_dataset and supervised_weight > 0:
            n_sup = int(len(supervised_dataset) * supervised_weight)
            datasets.append(torch.utils.data.Subset(supervised_dataset, range(n_sup)))
        
        if not datasets:
            raise ValueError("At least one dataset must be provided with weight > 0")
        
        super().__init__(datasets)


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


class StreamingMixedDataset(torch.utils.data.IterableDataset):
    """Stream a mix of unsupervised and supervised tokenized data."""

    def __init__(
        self,
        unsup_paths: List[str],
        sup_paths: List[str],
        unsup_fraction: float = 0.5,
        max_samples: Optional[int] = None,
    ):
        if not (0 <= unsup_fraction <= 1):
            raise ValueError("unsup_fraction must be between 0 and 1")
        self.unsup_dataset = StreamingTokenizedDataset(unsup_paths, with_labels=False) if unsup_paths else None
        self.sup_dataset = StreamingTokenizedDataset(sup_paths, with_labels=False) if sup_paths else None
        if self.unsup_dataset is None and self.sup_dataset is None:
            raise ValueError("At least one stream (unsup or sup) must be provided")
        self.unsup_fraction = unsup_fraction
        self.max_samples = max_samples

    def __iter__(self):
        unsup_iter = iter(self.unsup_dataset) if self.unsup_dataset else None
        sup_iter = iter(self.sup_dataset) if self.sup_dataset else None
        total = 0
        while True:
            if self.max_samples and total >= self.max_samples:
                return

            source = random.random()
            if self.unsup_dataset and source < self.unsup_fraction:
                try:
                    yield next(unsup_iter)
                except StopIteration:
                    if sup_iter:
                        for item in sup_iter:
                            yield item
                            total += 1
                            if self.max_samples and total >= self.max_samples:
                                return
                    return
            else:
                if sup_iter is None:
                    if unsup_iter is None:
                        return
                    try:
                        yield next(unsup_iter)
                    except StopIteration:
                        return
                    total += 1
                    continue
                try:
                    yield next(sup_iter)
                except StopIteration:
                    for item in unsup_iter:
                        yield item
                        total += 1
                        if self.max_samples and total >= self.max_samples:
                            return
                    return
            total += 1

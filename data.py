"""data.py"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Optional, Tuple
import pickle
from pathlib import Path


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


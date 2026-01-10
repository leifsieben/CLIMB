"""
evaluate_model.py
=================
Evaluate pretrained encoder on downstream tasks (MoleculeNet, drug discovery, etc.)

Supports:
- MoleculeNet datasets (BBBP, Tox21, ESOL, etc.)
- Custom datasets with SMILES + labels
- Multiple evaluation metrics
- Fine-tuning with frozen/unfrozen encoder

Usage:
    # Evaluate on MoleculeNet BBBP
    python evaluate_model.py \
        --pretrained_model models/mlm_100 \
        --dataset moleculenet \
        --dataset_name BBBP \
        --output results/bbbp_eval
    
    # Evaluate on custom dataset
    python evaluate_model.py \
        --pretrained_model models/mlm_100 \
        --dataset custom \
        --train_data data/custom_train.csv \
        --val_data data/custom_val.csv \
        --test_data data/custom_test.csv \
        --task_type classification \
        --output results/custom_eval
"""

import argparse
import logging
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
from utils import setup_logging, get_device

logger = logging.getLogger(__name__)


# ==============================================================================
# MoleculeNet Dataset Loader
# ==============================================================================

class MoleculeNetLoader:
    """
    Load MoleculeNet datasets with standard splits
    Uses DeepChem for easy access to MoleculeNet
    """
    
    DATASETS = {
        'BBBP': {'task': 'classification', 'num_tasks': 1, 'metric': 'roc_auc'},
        'Tox21': {'task': 'classification', 'num_tasks': 12, 'metric': 'roc_auc'},
        'ESOL': {'task': 'regression', 'num_tasks': 1, 'metric': 'rmse'},
        'FreeSolv': {'task': 'regression', 'num_tasks': 1, 'metric': 'rmse'},
        'Lipophilicity': {'task': 'regression', 'num_tasks': 1, 'metric': 'rmse'},
        'BACE': {'task': 'classification', 'num_tasks': 1, 'metric': 'roc_auc'},
        'SIDER': {'task': 'classification', 'num_tasks': 27, 'metric': 'roc_auc'},
        'ClinTox': {'task': 'classification', 'num_tasks': 2, 'metric': 'roc_auc'},
    }
    
    def __init__(self, dataset_name: str, split_type: str = 'scaffold'):
        """
        Initialize MoleculeNet loader
        
        Args:
            dataset_name: Name of MoleculeNet dataset (e.g., 'BBBP')
            split_type: 'scaffold', 'random', or 'stratified'
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.info = self.DATASETS[dataset_name]
        
        logger.info(f"Loading MoleculeNet dataset: {dataset_name}")
        logger.info(f"  Task: {self.info['task']}")
        logger.info(f"  Num tasks: {self.info['num_tasks']}")
        logger.info(f"  Split: {split_type}")
    
    def load(self):
        """
        Load dataset with train/val/test splits
        
        Returns:
            Dictionary with keys: train, val, test
            Each containing: smiles (list), labels (array), valid_mask (array for missing values)
        """
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError(
                "DeepChem required for MoleculeNet datasets. "
                "Install with: pip install deepchem"
            )
        
        # Load dataset using DeepChem
        # Use 'raw' featurizer to preserve SMILES - we'll tokenize with our own tokenizer
        logger.info("Loading via DeepChem...")

        if self.dataset_name == 'BBBP':
            tasks, datasets, transformers = dc.molnet.load_bbbp(
                featurizer='raw',
                splitter=self.split_type
            )
        elif self.dataset_name == 'Tox21':
            tasks, datasets, transformers = dc.molnet.load_tox21(
                featurizer='raw',
                splitter=self.split_type
            )
        elif self.dataset_name == 'ESOL':
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer='raw',
                splitter=self.split_type
            )
        elif self.dataset_name == 'FreeSolv':
            tasks, datasets, transformers = dc.molnet.load_sampl(
                featurizer='raw',
                splitter=self.split_type
            )
        elif self.dataset_name == 'Lipophilicity':
            tasks, datasets, transformers = dc.molnet.load_lipo(
                featurizer='raw',
                splitter=self.split_type
            )
        elif self.dataset_name == 'BACE':
            tasks, datasets, transformers = dc.molnet.load_bace_classification(
                featurizer='raw',
                splitter=self.split_type
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not yet implemented")
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # Extract SMILES and labels
        result = {}
        for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            smiles = dataset.ids  # SMILES strings
            labels = dataset.y    # Labels (may contain NaN for multi-task)
            weights = dataset.w   # Valid mask (1 if valid, 0 if missing)
            
            result[split_name] = {
                'smiles': list(smiles),
                'labels': labels,
                'weights': weights,  # For handling missing values in multi-task
            }
            
            logger.info(f"  {split_name}: {len(smiles)} molecules")
        
        return result, tasks


class CustomDatasetLoader:
    """Load custom datasets from CSV files"""
    
    def __init__(self, task_type: str, num_tasks: int = 1):
        """
        Args:
            task_type: 'classification' or 'regression'
            num_tasks: Number of prediction tasks
        """
        self.task_type = task_type
        self.num_tasks = num_tasks
    
    def load(self, train_file: str, val_file: str, test_file: str):
        """
        Load custom dataset from CSV files
        
        Expected CSV format:
            SMILES,label1,label2,...
        
        Returns:
            Dictionary with train/val/test splits
        """
        result = {}
        
        for split_name, file_path in [('train', train_file), ('val', val_file), ('test', test_file)]:
            if not file_path:
                continue
                
            df = pd.read_csv(file_path)
            
            # Extract SMILES
            if 'SMILES' not in df.columns:
                raise ValueError(f"CSV must have 'SMILES' column. Found: {df.columns.tolist()}")
            
            smiles = df['SMILES'].tolist()
            
            # Extract labels (all columns except SMILES)
            label_cols = [col for col in df.columns if col != 'SMILES']
            labels = df[label_cols].values
            
            # Create weights (all 1s for custom datasets - no missing values assumed)
            weights = np.ones_like(labels)
            
            result[split_name] = {
                'smiles': smiles,
                'labels': labels,
                'weights': weights,
            }
            
            logger.info(f"  {split_name}: {len(smiles)} molecules")
        
        return result, label_cols


# ==============================================================================
# Dataset Class for Fine-tuning
# ==============================================================================

class DownstreamDataset(Dataset):
    """Dataset for fine-tuning on downstream tasks"""

    def __init__(self, tokenized_data, labels, weights=None, task_type='regression'):
        self.data = tokenized_data
        self.labels = labels
        self.weights = weights if weights is not None else np.ones_like(labels)
        self.task_type = task_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        label = self.labels[idx]

        if self.task_type == 'classification':
            # CrossEntropyLoss expects long tensor of shape (batch_size,)
            # Squeeze if shape is (1,) and convert to int
            if hasattr(label, '__len__') and len(label) == 1:
                label = label[0]
            item['labels'] = torch.tensor(int(label), dtype=torch.long)
        else:
            # Regression: float tensor
            item['labels'] = torch.tensor(label, dtype=torch.float)

        return item


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def compute_metrics_classification(predictions, labels, weights=None):
    """Compute metrics for classification tasks"""
    metrics = {}
    
    # Convert to numpy
    preds = predictions
    labels = labels
    
    if weights is not None:
        # Handle multi-task with missing values
        valid_mask = weights.astype(bool)
    else:
        valid_mask = np.ones_like(labels, dtype=bool)
    
    # For each task
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        # Single task
        preds_valid = preds[valid_mask.flatten()]
        labels_valid = labels[valid_mask.flatten()]
        
        if len(np.unique(labels_valid)) > 1:  # Need both classes
            metrics['roc_auc'] = roc_auc_score(labels_valid, preds_valid)
            metrics['avg_precision'] = average_precision_score(labels_valid, preds_valid)
    else:
        # Multi-task
        task_aucs = []
        for i in range(labels.shape[1]):
            mask = valid_mask[:, i]
            if mask.sum() > 0 and len(np.unique(labels[mask, i])) > 1:
                auc = roc_auc_score(labels[mask, i], preds[mask, i])
                task_aucs.append(auc)
        
        if task_aucs:
            metrics['roc_auc'] = np.mean(task_aucs)
            metrics['roc_auc_per_task'] = task_aucs
    
    return metrics


def compute_metrics_regression(predictions, labels, weights=None):
    """Compute metrics for regression tasks"""
    metrics = {}
    
    if weights is not None:
        valid_mask = weights.astype(bool).flatten()
        preds = predictions[valid_mask]
        labels = labels[valid_mask]
    else:
        preds = predictions
        labels = labels
    
    metrics['rmse'] = np.sqrt(mean_squared_error(labels, preds))
    metrics['mae'] = np.mean(np.abs(labels - preds))
    metrics['r2'] = r2_score(labels, preds)
    
    return metrics


# ==============================================================================
# Main Evaluation Function
# ==============================================================================

def evaluate_on_dataset(
    pretrained_model_path: str,
    dataset_splits: dict,
    task_type: str,
    num_tasks: int,
    output_dir: str,
    tokenizer_path: str = None,
    freeze_encoder: bool = False,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune pretrained encoder on downstream task and evaluate
    
    Args:
        pretrained_model_path: Path to pretrained model directory
        dataset_splits: Dict with 'train', 'val', 'test' data
        task_type: 'classification' or 'regression'
        num_tasks: Number of prediction tasks
        output_dir: Where to save results
        tokenizer_path: Path to tokenizer (if different from model path)
        freeze_encoder: If True, only train the classification head
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate for fine-tuning
    """
    
    logger.info("="*80)
    logger.info("DOWNSTREAM TASK EVALUATION")
    logger.info("="*80)
    logger.info(f"Pretrained model: {pretrained_model_path}")
    logger.info(f"Task type: {task_type}")
    logger.info(f"Num tasks: {num_tasks}")
    logger.info(f"Freeze encoder: {freeze_encoder}")
    
    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = pretrained_model_path
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tokenizer_path) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    
    # Tokenize data
    tokenized_splits = {}
    for split_name, split_data in dataset_splits.items():
        logger.info(f"Tokenizing {split_name} split...")
        tokenized = []
        for smiles in split_data['smiles']:
            encoding = tokenizer(
                smiles,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None,
            )
            tokenized.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            })
        
        tokenized_splits[split_name] = DownstreamDataset(
            tokenized,
            split_data['labels'],
            split_data.get('weights'),
            task_type=task_type,
        )
    
    # Load pretrained model and add classification head
    logger.info("Loading pretrained encoder...")

    # For classification: num_labels=2 for binary (CrossEntropyLoss expects 2 classes)
    # For regression: num_labels=num_tasks (number of output values)
    if task_type == "classification":
        model_num_labels = 2  # Binary classification
    else:
        model_num_labels = num_tasks

    model = RobertaForSequenceClassification.from_pretrained(
        pretrained_model_path,
        num_labels=model_num_labels,
        problem_type="single_label_classification" if task_type == "classification" else "regression",
        ignore_mismatched_sizes=True,  # Ignore classification head size mismatch
    )
    
    # Freeze encoder if requested
    if freeze_encoder:
        logger.info("Freezing encoder layers...")
        for param in model.roberta.parameters():
            param.requires_grad = False
        
        # Only train the classifier
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable parameters: {n_trainable:,}")
    
    # Training arguments - let Transformers auto-detect device (CUDA > MPS > CPU)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits['train'],
        eval_dataset=tokenized_splits.get('val'),
        data_collator=data_collator,
    )
    
    # Fine-tune
    logger.info("Fine-tuning on downstream task...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = trainer.predict(tokenized_splits['test'])
    
    # Compute metrics
    if task_type == "classification":
        # Apply softmax and get probability of class 1 for binary classification
        logits = torch.tensor(test_predictions.predictions)
        probs = torch.softmax(logits, dim=-1)
        preds = probs[:, 1].numpy()  # Probability of positive class
        metrics = compute_metrics_classification(
            preds,
            dataset_splits['test']['labels'],
            dataset_splits['test'].get('weights'),
        )
    else:
        preds = test_predictions.predictions
        metrics = compute_metrics_regression(
            preds,
            dataset_splits['test']['labels'],
            dataset_splits['test'].get('weights'),
        )
    
    # Define random baselines for comparison
    if task_type == "classification":
        baselines = {
            'roc_auc': 0.5,  # Random classifier
            'avg_precision': 0.5,  # Approximate for balanced data
        }
    else:
        baselines = {
            'rmse': None,  # Depends on data scale
            'mae': None,
            'r2': 0.0,  # Random/mean predictor
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build human-readable summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("EVALUATION RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Model:          {pretrained_model_path}")
    summary_lines.append(f"Task Type:      {task_type}")
    summary_lines.append(f"Num Tasks:      {num_tasks}")
    summary_lines.append(f"Freeze Encoder: {freeze_encoder}")
    summary_lines.append(f"Train Samples:  {len(tokenized_splits['train'])}")
    summary_lines.append(f"Test Samples:   {len(tokenized_splits['test'])}")
    summary_lines.append("")
    summary_lines.append("-" * 80)
    summary_lines.append("TEST SET METRICS")
    summary_lines.append("-" * 80)
    summary_lines.append("")
    summary_lines.append(f"{'Metric':<20} {'Value':>12} {'Random Baseline':>18} {'vs Baseline':>15}")
    summary_lines.append("-" * 65)

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float, np.floating)):
            baseline = baselines.get(metric_name)
            if baseline is not None:
                if metric_name == 'rmse' or metric_name == 'mae':
                    # Lower is better
                    diff = baseline - metric_value
                    comparison = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                else:
                    # Higher is better
                    diff = metric_value - baseline
                    comparison = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                summary_lines.append(f"{metric_name:<20} {metric_value:>12.4f} {baseline:>18.4f} {comparison:>15}")
            else:
                summary_lines.append(f"{metric_name:<20} {metric_value:>12.4f} {'N/A':>18} {'N/A':>15}")
        else:
            summary_lines.append(f"{metric_name:<20} {str(metric_value)[:12]:>12}")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    # Log summary
    for line in summary_lines:
        logger.info(line)

    # Save summary to TXT file
    summary_text = "\n".join(summary_lines)
    with open(output_path / "results_summary.txt", 'w') as f:
        f.write(summary_text)

    # Save results to JSON
    results = {
        'pretrained_model': pretrained_model_path,
        'task_type': task_type,
        'num_tasks': num_tasks,
        'freeze_encoder': freeze_encoder,
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()},
        'baselines': {k: v for k, v in baselines.items() if v is not None},
        'num_train': len(tokenized_splits['train']),
        'num_test': len(tokenized_splits['test']),
    }

    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    np.save(output_path / "test_predictions.npy", preds)
    np.save(output_path / "test_labels.npy", dataset_splits['test']['labels'])

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - results_summary.txt (human-readable)")
    logger.info(f"  - results.json (machine-readable)")
    logger.info(f"  - test_predictions.npy")
    logger.info(f"  - test_labels.npy")

    return metrics


# ==============================================================================
# Main Script
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained model on downstream tasks")
    parser.add_argument("--pretrained_model", required=True, help="Path to pretrained model")
    parser.add_argument("--dataset", choices=["moleculenet", "custom"], required=True)
    
    # MoleculeNet options
    parser.add_argument("--dataset_name", help="MoleculeNet dataset name (BBBP, Tox21, etc.)")
    parser.add_argument("--split_type", default="scaffold", choices=["scaffold", "random", "stratified"])
    
    # Custom dataset options
    parser.add_argument("--train_data", help="Training data CSV")
    parser.add_argument("--val_data", help="Validation data CSV")
    parser.add_argument("--test_data", help="Test data CSV")
    parser.add_argument("--task_type", choices=["classification", "regression"])
    parser.add_argument("--num_tasks", type=int, default=1)
    
    # Training options
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tokenizer", help="Tokenizer path (if different from model)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder during fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--log_file", help="Log file path")
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    # Load dataset
    if args.dataset == "moleculenet":
        if not args.dataset_name:
            raise ValueError("--dataset_name required for MoleculeNet")
        
        loader = MoleculeNetLoader(args.dataset_name, args.split_type)
        dataset_splits, tasks = loader.load()
        task_type = loader.info['task']
        num_tasks = loader.info['num_tasks']
        
    else:  # custom
        if not all([args.train_data, args.test_data, args.task_type]):
            raise ValueError("--train_data, --test_data, --task_type required for custom dataset")
        
        loader = CustomDatasetLoader(args.task_type, args.num_tasks)
        dataset_splits, tasks = loader.load(args.train_data, args.val_data, args.test_data)
        task_type = args.task_type
        num_tasks = args.num_tasks
    
    # Evaluate
    metrics = evaluate_on_dataset(
        pretrained_model_path=args.pretrained_model,
        dataset_splits=dataset_splits,
        task_type=task_type,
        num_tasks=num_tasks,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        freeze_encoder=args.freeze_encoder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
"""
evaluate_model.py
=================
Evaluate pretrained encoder on downstream tasks (MoleculeNet, drug discovery, etc.)

Supports:
- MoleculeNet datasets (BBBP, Tox21, ESOL, etc.)
- Custom datasets with SMILES + labels
- Multiple evaluation metrics
- Fine-tuning with frozen/unfrozen encoder
- Multi-task model evaluation (from train_multitask.py)

Usage:
    # Evaluate encoder on MoleculeNet BBBP
    python evaluate_model.py \
        --pretrained_model models/mlm_100 \
        --dataset moleculenet \
        --dataset_name BBBP \
        --output results/bbbp_eval

    # Evaluate multi-task model on specific task
    python evaluate_model.py \
        --pretrained_model experiments/multitask/final \
        --dataset moleculenet \
        --dataset_name BBBP \
        --output results/bbbp_eval \
        --multitask_model \
        --task_name BBBP

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


# Optional multi-task imports (only needed for --multitask_model)
def _import_multitask():
    """Lazy import multi-task components."""
    from multitask_model import MultiTaskModel
    from tasks import TaskRegistry
    return MultiTaskModel, TaskRegistry


# ==============================================================================
# MoleculeNet Dataset Loader
# ==============================================================================

class MoleculeNetLoader:
    """
    Load MoleculeNet datasets with standard splits via DeepChem.
    Supports the full MoleculeNet suite.
    """

    DATASETS = {
        # Classification
        'BBBP': {'task': 'classification', 'metric': 'roc_auc'},
        'Tox21': {'task': 'classification', 'metric': 'roc_auc'},
        'ToxCast': {'task': 'classification', 'metric': 'roc_auc'},
        'SIDER': {'task': 'classification', 'metric': 'roc_auc'},
        'ClinTox': {'task': 'classification', 'metric': 'roc_auc'},
        'HIV': {'task': 'classification', 'metric': 'roc_auc'},
        'BACE': {'task': 'classification', 'metric': 'roc_auc'},
        'MUV': {'task': 'classification', 'metric': 'roc_auc'},
        'PCBA': {'task': 'classification', 'metric': 'roc_auc'},
        # Regression
        'ESOL': {'task': 'regression', 'metric': 'rmse'},
        'FreeSolv': {'task': 'regression', 'metric': 'rmse'},
        'Lipophilicity': {'task': 'regression', 'metric': 'rmse'},
        'QM7': {'task': 'regression', 'metric': 'rmse'},
        'QM8': {'task': 'regression', 'metric': 'rmse'},
        'QM9': {'task': 'regression', 'metric': 'rmse'},
    }

    def __init__(self, dataset_name: str, split_type: str = 'scaffold'):
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")

        self.dataset_name = dataset_name
        self.split_type = split_type
        self.info = self.DATASETS[dataset_name]

        logger.info(f"Loading MoleculeNet dataset: {dataset_name}")
        logger.info(f"  Task: {self.info['task']}")
        logger.info(f"  Split: {split_type}")

    def _load_dataset(self):
        """Load dataset using DeepChem with raw featurizer."""
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError(
                "DeepChem required for MoleculeNet datasets. "
                "Install with: pip install deepchem"
            )

        name = self.dataset_name
        splitter = self.split_type

        if name == 'BBBP':
            return dc.molnet.load_bbbp(featurizer='raw', splitter=splitter)
        if name == 'Tox21':
            return dc.molnet.load_tox21(featurizer='raw', splitter=splitter)
        if name == 'ToxCast':
            return dc.molnet.load_toxcast(featurizer='raw', splitter=splitter)
        if name == 'SIDER':
            return dc.molnet.load_sider(featurizer='raw', splitter=splitter)
        if name == 'ClinTox':
            return dc.molnet.load_clintox(featurizer='raw', splitter=splitter)
        if name == 'HIV':
            return dc.molnet.load_hiv(featurizer='raw', splitter=splitter)
        if name == 'BACE':
            return dc.molnet.load_bace_classification(featurizer='raw', splitter=splitter)
        if name == 'MUV':
            return dc.molnet.load_muv(featurizer='raw', splitter=splitter)
        if name == 'PCBA':
            return dc.molnet.load_pcba(featurizer='raw', splitter=splitter)
        if name == 'ESOL':
            return dc.molnet.load_delaney(featurizer='raw', splitter=splitter)
        if name == 'FreeSolv':
            return dc.molnet.load_sampl(featurizer='raw', splitter=splitter)
        if name == 'Lipophilicity':
            return dc.molnet.load_lipo(featurizer='raw', splitter=splitter)
        if name == 'QM7':
            return dc.molnet.load_qm7(featurizer='raw', splitter=splitter)
        if name == 'QM8':
            return dc.molnet.load_qm8(featurizer='raw', splitter=splitter)
        if name == 'QM9':
            return dc.molnet.load_qm9(featurizer='raw', splitter=splitter)

        raise NotImplementedError(f"Loader not implemented for {name}")

    def load(self):
        """Load dataset with train/val/test splits."""
        tasks, datasets, transformers = self._load_dataset()
        train_dataset, val_dataset, test_dataset = datasets

        # Update num_tasks dynamically
        num_tasks = len(tasks) if isinstance(tasks, (list, tuple)) else 1
        self.info['num_tasks'] = num_tasks

        result = {}
        for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            smiles = dataset.ids
            labels = dataset.y
            weights = dataset.w

            result[split_name] = {
                'smiles': list(smiles),
                'labels': labels,
                'weights': weights,
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
            # Single-task: int label; multi-task: vector of floats (multi-label)
            if hasattr(label, '__len__') and len(label) > 1:
                item['labels'] = torch.tensor(label, dtype=torch.float32)
            else:
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
    if preds.ndim == 1 or labels.ndim == 1 or (labels.ndim > 1 and labels.shape[1] == 1):
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

    preds = predictions
    lbls = labels
    mask = weights.astype(bool) if weights is not None else None

    if preds.ndim == 1 or lbls.ndim == 1:
        if mask is not None:
            mask_flat = mask.flatten()
            preds = preds[mask_flat]
            lbls = lbls[mask_flat]
    else:
        # Multi-task regression
        task_metrics = []
        for i in range(lbls.shape[1]):
            task_pred = preds[:, i]
            task_lbl = lbls[:, i]
            if mask is not None:
                task_mask = mask[:, i].astype(bool)
                if task_mask.sum() == 0:
                    continue
                task_pred = task_pred[task_mask]
                task_lbl = task_lbl[task_mask]
            task_metrics.append({
                'rmse': np.sqrt(mean_squared_error(task_lbl, task_pred)),
                'mae': np.mean(np.abs(task_lbl - task_pred)),
                'r2': r2_score(task_lbl, task_pred),
            })

        if task_metrics:
            metrics['rmse'] = float(np.mean([m['rmse'] for m in task_metrics]))
            metrics['mae'] = float(np.mean([m['mae'] for m in task_metrics]))
            metrics['r2'] = float(np.mean([m['r2'] for m in task_metrics]))
            metrics['rmse_per_task'] = [m['rmse'] for m in task_metrics]
            metrics['mae_per_task'] = [m['mae'] for m in task_metrics]
            metrics['r2_per_task'] = [m['r2'] for m in task_metrics]
        return metrics

    # Single-task or flattened case
    metrics['rmse'] = np.sqrt(mean_squared_error(lbls, preds))
    metrics['mae'] = np.mean(np.abs(lbls - preds))
    metrics['r2'] = r2_score(lbls, preds)
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
    dataset_name: str = "",
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

    # For classification: num_labels=2 for binary or num_tasks for multi-label
    # For regression: num_labels=num_tasks (number of output values)
    if task_type == "classification":
        if num_tasks > 1:
            model_num_labels = num_tasks
            problem_type = "multi_label_classification"
        else:
            model_num_labels = 2  # Binary classification
            problem_type = "single_label_classification"
    else:
        model_num_labels = num_tasks
        problem_type = "regression"

    model = RobertaForSequenceClassification.from_pretrained(
        pretrained_model_path,
        num_labels=model_num_labels,
        problem_type=problem_type,
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
        remove_unused_columns=False,
    )
    
    # Custom collator to support multi-label classification tensors
    def collate_fn(batch):
        labels = [b.pop('labels') for b in batch]
        labels = torch.stack(labels) if torch.is_tensor(labels[0]) else torch.tensor(labels)
        padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        padded['labels'] = labels
        return padded
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits['train'],
        eval_dataset=tokenized_splits.get('val'),
        data_collator=collate_fn,
    )
    
    # Fine-tune
    logger.info("Fine-tuning on downstream task...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = trainer.predict(tokenized_splits['test'])

    # Compute metrics
    test_labels = np.asarray(dataset_splits['test']['labels'])
    test_weights = dataset_splits['test'].get('weights')
    if test_weights is not None:
        test_weights = np.asarray(test_weights)
    # Squeeze trailing singleton task dim for single-task cases
    if test_labels.ndim == 2 and test_labels.shape[1] == 1:
        test_labels = test_labels.squeeze(1)
        if test_weights is not None and test_weights.ndim == 2 and test_weights.shape[1] == 1:
            test_weights = test_weights.squeeze(1)

    if task_type == "classification":
        logits = torch.tensor(test_predictions.predictions)
        if num_tasks > 1:
            probs = torch.sigmoid(logits)
            preds = probs.numpy()  # [N, num_tasks]
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = probs[:, 1].numpy()  # Probability of positive class
        metrics = compute_metrics_classification(
            preds,
            test_labels,
            test_weights,
        )
    else:
        preds = test_predictions.predictions
        metrics = compute_metrics_regression(
            preds,
            test_labels,
            test_weights,
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
    summary_lines.append(f"Dataset:        {dataset_name or 'custom'}")
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
        'dataset': dataset_name or 'custom',
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

    # Append to central aggregate log (human-readable)
    aggregate_path = output_path.parent / "all_evaluations.txt"
    main_metric = 'roc_auc' if task_type == 'classification' else 'rmse'
    metric_value = metrics.get(main_metric)
    baseline_value = baselines.get(main_metric) if main_metric in baselines else None
    with open(aggregate_path, 'a') as agg:
        agg.write(
            f"{dataset_name or 'custom'}\t{task_type}\t{main_metric}\t"
            f"{metric_value if metric_value is not None else 'NA'}\t"
            f"baseline={baseline_value if baseline_value is not None else 'NA'}\t"
            f"model={pretrained_model_path}\n"
        )
    logger.info(f"Appended aggregate metrics to {aggregate_path}")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - results_summary.txt (human-readable)")
    logger.info(f"  - results.json (machine-readable)")
    logger.info(f"  - test_predictions.npy")
    logger.info(f"  - test_labels.npy")

    return metrics


# ==============================================================================
# Multi-Task Model Evaluation
# ==============================================================================

def evaluate_multitask_model(
    model_path: str,
    dataset_splits: dict,
    task_name: str,
    task_type: str,
    output_dir: str,
    tokenizer_path: str = None,
    batch_size: int = 32,
):
    """
    Evaluate a trained MultiTaskModel on a specific task.

    No fine-tuning - just inference with the trained task head.

    Args:
        model_path: Path to MultiTaskModel checkpoint
        dataset_splits: Dict with 'test' data
        task_name: Name of task to evaluate
        task_type: 'classification' or 'regression'
        output_dir: Where to save results
        tokenizer_path: Path to tokenizer
        batch_size: Batch size for inference
    """
    MultiTaskModel, TaskRegistry = _import_multitask()

    logger.info("=" * 80)
    logger.info("MULTI-TASK MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Task: {task_name}")

    # Load model
    model = MultiTaskModel.from_checkpoint(model_path)

    if task_name not in model.task_heads:
        raise ValueError(
            f"Task '{task_name}' not in model. Available: {list(model.task_heads.keys())}"
        )

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = model_path

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tokenizer_path) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    # Tokenize test data
    test_data = dataset_splits['test']
    logger.info(f"Tokenizing {len(test_data['smiles'])} test samples...")

    tokenized = tokenizer(
        test_data['smiles'],
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors='pt',
    )

    # Move to device
    device = get_device()
    model.to(device)
    model.eval()

    # Inference
    logger.info("Running inference...")
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(test_data['smiles']), batch_size):
            batch_input_ids = tokenized['input_ids'][i:i+batch_size].to(device)
            batch_attention_mask = tokenized['attention_mask'][i:i+batch_size].to(device)

            outputs = model.forward_supervised(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            logits = outputs['logits'][task_name]

            if task_type == 'classification':
                probs = torch.softmax(logits, dim=-1)
                preds = probs[:, 1].cpu().numpy()  # Prob of positive class
            else:
                preds = logits.squeeze(-1).cpu().numpy()

            all_preds.extend(preds)

    preds = np.array(all_preds)

    # Compute metrics
    if task_type == "classification":
        metrics = compute_metrics_classification(
            preds,
            test_data['labels'],
            test_data.get('weights'),
        )
        baselines = {'roc_auc': 0.5, 'avg_precision': 0.5}
    else:
        metrics = compute_metrics_regression(
            preds,
            test_data['labels'],
            test_data.get('weights'),
        )
        baselines = {'rmse': None, 'mae': None, 'r2': 0.0}

    # Save results (same format as single-task)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("MULTI-TASK MODEL EVALUATION RESULTS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Model:          {model_path}")
    summary_lines.append(f"Task:           {task_name}")
    summary_lines.append(f"Task Type:      {task_type}")
    summary_lines.append(f"Test Samples:   {len(test_data['smiles'])}")
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
                diff = metric_value - baseline
                comparison = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                summary_lines.append(
                    f"{metric_name:<20} {metric_value:>12.4f} {baseline:>18.4f} {comparison:>15}"
                )
            else:
                summary_lines.append(
                    f"{metric_name:<20} {metric_value:>12.4f} {'N/A':>18} {'N/A':>15}"
                )

    summary_lines.append("")
    summary_lines.append("=" * 80)

    for line in summary_lines:
        logger.info(line)

    with open(output_path / "results_summary.txt", 'w') as f:
        f.write("\n".join(summary_lines))

    results = {
        'model_path': model_path,
        'task_name': task_name,
        'task_type': task_type,
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()},
        'num_test': len(test_data['smiles']),
    }

    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    np.save(output_path / "test_predictions.npy", preds)
    np.save(output_path / "test_labels.npy", test_data['labels'])

    logger.info(f"\nResults saved to: {output_dir}")

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

    # Multi-task model options
    parser.add_argument("--multitask_model", action="store_true",
                       help="Evaluate a MultiTaskModel (no fine-tuning, just inference)")
    parser.add_argument("--task_name", help="Task name for multi-task model evaluation")

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
    if args.multitask_model:
        # Evaluate pre-trained MultiTaskModel (no fine-tuning)
        task_name = args.task_name or args.dataset_name
        if not task_name:
            raise ValueError("--task_name required for --multitask_model")

        metrics = evaluate_multitask_model(
            model_path=args.pretrained_model,
            dataset_splits=dataset_splits,
            task_name=task_name,
            task_type=task_type,
            output_dir=args.output,
            tokenizer_path=args.tokenizer,
            batch_size=args.batch_size,
        )
    else:
        # Standard evaluation: fine-tune encoder on downstream task
        metrics = evaluate_on_dataset(
            pretrained_model_path=args.pretrained_model,
            dataset_splits=dataset_splits,
            task_type=task_type,
            num_tasks=num_tasks,
            output_dir=args.output,
            dataset_name=args.dataset_name or "custom",
            tokenizer_path=args.tokenizer,
            freeze_encoder=args.freeze_encoder,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()

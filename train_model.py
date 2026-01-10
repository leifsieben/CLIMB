# ==============================================================================
# FILE: train_model.py
# Flexible training with support for any mixture of supervised/unsupervised
# ==============================================================================

"""train_model.py

Flexible training supporting:
- 100% unsupervised (MLM): --unsup_weight 1.0 --sup_weight 0.0
- 100% supervised: --unsup_weight 0.0 --sup_weight 1.0
- Mixed: --unsup_weight 0.5 --sup_weight 0.5

Usage:
    # 100% Unsupervised (MLM)
    python train_model.py \
        --config config.yaml \
        --tokenizer tokenizer/ \
        --unsup_data data/unsup.pkl \
        --unsup_weight 1.0 \
        --sup_weight 0.0 \
        --output models/mlm_100 \
        --task mlm
    
    # 50/50 Mixed (MLM on both)
    python train_model.py \
        --config config.yaml \
        --tokenizer tokenizer/ \
        --unsup_data data/unsup.pkl \
        --sup_data data/sup.pkl \
        --unsup_weight 0.5 \
        --sup_weight 0.5 \
        --output models/mlm_50_50 \
        --task mlm
    
    # 100% Supervised (Regression)
    python train_model.py \
        --config config.yaml \
        --tokenizer tokenizer/ \
        --sup_data data/sup.pkl \
        --unsup_weight 0.0 \
        --sup_weight 1.0 \
        --output models/supervised_100 \
        --task regression
"""

import argparse
import logging
import pickle
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, Subset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
from utils import setup_logging, load_config, get_device
from model import create_model

logger = logging.getLogger(__name__)


# Dataset classes
class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class SupervisedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# Model creation now handled by model.py


def create_mixed_dataset(
    unsup_data: dict = None,
    sup_data: dict = None,
    unsup_weight: float = 1.0,
    sup_weight: float = 0.0,
    task: str = "mlm",
):
    """
    Create mixed dataset from unsupervised and supervised data
    
    Args:
        unsup_data: Dictionary with 'data' key
        sup_data: Dictionary with 'data' and 'labels' keys
        unsup_weight: Weight for unsupervised data (0.0 to 1.0)
        sup_weight: Weight for supervised data (0.0 to 1.0)
        task: 'mlm' or 'regression'
    
    Returns:
        Combined dataset
    """
    
    if unsup_weight == 0.0 and sup_weight == 0.0:
        raise ValueError("At least one weight must be > 0")
    
    datasets = []
    
    # Add unsupervised data
    if unsup_data and unsup_weight > 0:
        unsup_dataset = UnsupervisedDataset(unsup_data['data'])
        n_unsup = int(len(unsup_dataset) * unsup_weight)
        if n_unsup > 0:
            datasets.append(Subset(unsup_dataset, range(n_unsup)))
            logger.info(f"Added {n_unsup} unsupervised samples ({unsup_weight*100:.0f}%)")
    
    # Add supervised data
    if sup_data and sup_weight > 0:
        if task == "mlm":
            # For MLM, treat supervised data as unsupervised (ignore labels)
            sup_dataset = UnsupervisedDataset(sup_data['data'])
        else:
            # For regression, use labels
            sup_dataset = SupervisedDataset(sup_data['data'], sup_data['labels'])
        
        n_sup = int(len(sup_dataset) * sup_weight)
        if n_sup > 0:
            datasets.append(Subset(sup_dataset, range(n_sup)))
            logger.info(f"Added {n_sup} supervised samples ({sup_weight*100:.0f}%)")
    
    if not datasets:
        raise ValueError("No data to train on!")
    
    # Combine datasets
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = ConcatDataset(datasets)
    
    logger.info(f"Total dataset size: {len(combined)}")
    
    return combined


def train(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    task: str,
    config: dict,
):
    """Train model"""
    
    # Data collator
    if task == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config['training'].get('mlm_probability', 0.15),
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - let Transformers auto-detect device (CUDA > MPS > CPU)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training'].get('num_epochs', 10),
        per_device_train_batch_size=config['training'].get('batch_size', 16),
        learning_rate=float(config['training'].get('learning_rate', 5e-5)),
        warmup_steps=config['training'].get('warmup_steps', 100),
        weight_decay=float(config['training'].get('weight_decay', 0.01)),
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['training'].get('logging_steps', 50),
        save_steps=config['training'].get('save_steps', 500),
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False if task == "mlm" else True,
    )

    logger.info(f"Training configuration:")
    logger.info(f"  Device: {get_device()}")
    logger.info(f"  Samples: {len(dataset)}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✓ Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train chemical language model with flexible data mixing")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    parser.add_argument("--unsup_data", help="Unsupervised data pickle")
    parser.add_argument("--sup_data", help="Supervised data pickle")
    parser.add_argument("--unsup_weight", type=float, default=1.0, help="Unsupervised data weight (0.0-1.0)")
    parser.add_argument("--sup_weight", type=float, default=0.0, help="Supervised data weight (0.0-1.0)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--task", choices=["mlm", "regression"], required=True, help="Training task")
    parser.add_argument("--log_file", help="Log file path")
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    logger.info("="*80)
    logger.info("CHEMICAL LANGUAGE MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Unsupervised weight: {args.unsup_weight}")
    logger.info(f"Supervised weight: {args.sup_weight}")
    
    # Validate inputs
    if args.unsup_weight == 0.0 and not args.sup_data:
        raise ValueError("Need supervised data if unsup_weight=0.0")
    if args.sup_weight == 0.0 and not args.unsup_data:
        raise ValueError("Need unsupervised data if sup_weight=0.0")
    
    # Load config
    config = load_config(args.config)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    
    # Load data
    unsup_data = None
    sup_data = None
    
    if args.unsup_data:
        with open(args.unsup_data, 'rb') as f:
            unsup_data = pickle.load(f)
        # Handle both formats: list or dict with 'data' key
        if not isinstance(unsup_data, dict):
            unsup_data = {'data': unsup_data}
        logger.info(f"Loaded unsupervised data: {len(unsup_data['data'])} samples")
    
    if args.sup_data:
        with open(args.sup_data, 'rb') as f:
            sup_data = pickle.load(f)
        logger.info(f"Loaded supervised data: {len(sup_data['data'])} samples")
    
    # Create mixed dataset
    dataset = create_mixed_dataset(
        unsup_data=unsup_data,
        sup_data=sup_data,
        unsup_weight=args.unsup_weight,
        sup_weight=args.sup_weight,
        task=args.task,
    )
    
    # Create model using model.py
    if args.task == "regression" and sup_data:
        num_labels = sup_data['labels'].shape[1]
    else:
        num_labels = 1
    
    model = create_model(
        vocab_size=len(tokenizer),
        task=args.task,
        num_labels=num_labels,
        **config['model']  # Unpack all model config from YAML
    )
    
    # Train
    train(model, tokenizer, dataset, args.output, args.task, config)
    
    # Save experiment config
    experiment_config = {
        'task': args.task,
        'unsup_weight': args.unsup_weight,
        'sup_weight': args.sup_weight,
        'model_config': config['model'],
        'training_config': config['training'],
    }
    
    config_file = Path(args.output) / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    logger.info(f"✓ Experiment config saved to {config_file}")
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
# ==============================================================================
# FILE: hyperparameter_search.py
# Find optimal hyperparameters using Optuna
# ==============================================================================

"""hyperparameter_search.py

Strategy: Run HP search on unsupervised (MLM) data for generalizability.
Use found hyperparameters for all subsequent experiments.

Usage:
    python hyperparameter_search.py \
        --config config.yaml \
        --tokenizer tokenizer/ \
        --train_data data/unsup.pkl \
        --eval_data data/unsup_eval.pkl \
        --output hp_search_results/ \
        --n_trials 20
"""

import argparse
import logging
import pickle
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, random_split
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
import optuna
from utils import setup_logging, load_config, get_device
from model import create_model

logger = logging.getLogger(__name__)


class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def objective(trial, tokenizer, train_dataset, eval_dataset, base_config):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    num_layers = trial.suggest_int("num_hidden_layers", 4, 12)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    
    # Attention heads: must divide hidden_size evenly
    # Standard ratios: hidden_size / head_dim, where head_dim is typically 64 or 32
    if hidden_size == 128:
        num_heads = 4  # 128/4 = 32 per head (or 2 heads with 64 each)
    elif hidden_size == 256:
        num_heads = trial.suggest_categorical("num_attention_heads_256", [4, 8])
    else:  # 512
        num_heads = trial.suggest_categorical("num_attention_heads_512", [8, 16])
    
    logger.info(f"\nTrial {trial.number}")
    logger.info(f"  LR: {learning_rate:.2e}, Batch: {batch_size}, Warmup: {warmup_steps}")
    logger.info(f"  Hidden: {hidden_size}, Layers: {num_layers}, Heads: {num_heads}, Weight decay: {weight_decay:.2e}")
    
    # Create model using model.py
    model = create_model(
        vocab_size=len(tokenizer),
        task="mlm",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=512,  # Fixed based on data analysis
        use_flash_attention=base_config.get('model', {}).get('use_flash_attention', True),
        use_gradient_checkpointing=False,  # Disable for HP search (faster)
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Training arguments - let Transformers auto-detect device (CUDA > MPS > CPU)
    output_dir = f"./hp_search_trial_{trial.number}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # More epochs for better convergence
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=10000,  # Don't save during HP search
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard during HP search
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    
    logger.info(f"  Trial {trial.number} eval loss: {eval_loss:.4f}")
    
    # Cleanup
    import shutil
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    
    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    parser.add_argument("--train_data", required=True, help="Training data pickle")
    parser.add_argument("--eval_data", help="Eval data pickle (optional, will split if not provided)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--log_file", help="Log file path")
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    logger.info("Starting hyperparameter search")
    logger.info(f"  Trials: {args.n_trials}")
    logger.info(f"  Device: {get_device()}")
    
    # Load config
    base_config = load_config(args.config)
    
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
    with open(args.train_data, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both formats: list or dict with 'data' key
    if isinstance(data, dict):
        tokenized_data = data['data']
    else:
        tokenized_data = data

    full_dataset = UnsupervisedDataset(tokenized_data)
    
    # Split train/eval if eval not provided
    if args.eval_data:
        with open(args.eval_data, 'rb') as f:
            eval_data = pickle.load(f)
        # Handle both formats
        if isinstance(eval_data, dict):
            eval_tokenized = eval_data['data']
        else:
            eval_tokenized = eval_data
        eval_dataset = UnsupervisedDataset(eval_tokenized)
        train_dataset = full_dataset
    else:
        train_size = int(0.9 * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Eval samples: {len(eval_dataset)}")
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, tokenizer, train_dataset, eval_dataset, base_config),
        n_trials=args.n_trials,
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER SEARCH COMPLETE")
    logger.info("="*80)
    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best eval loss: {study.best_value:.4f}")
    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best params as JSON
    best_params_file = output_path / "best_hyperparameters.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Save full study
    study_file = output_path / "optuna_study.pkl"
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    logger.info(f"\n✓ Results saved to {args.output}")
    logger.info(f"  Best params: {best_params_file}")
    logger.info(f"  Full study: {study_file}")


if __name__ == "__main__":
    main()


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
        learning_rate=config['training'].get('learning_rate', 5e-5),
        warmup_steps=config['training'].get('warmup_steps', 100),
        weight_decay=config['training'].get('weight_decay', 0.01),
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
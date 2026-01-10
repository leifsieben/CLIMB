# ==============================================================================
# FILE: train_multitask.py
# CLI entry point for multi-task supervised pretraining
# ==============================================================================

"""
Multi-task supervised pretraining.

Trains an encoder with multiple task heads on pooled datasets.
The encoder is the primary output (task heads are discarded).

Usage:
    python train_multitask.py --config config_multitask.yaml

Config Format (YAML):
    pretrained_model_path: ./experiments/mlm_pretrained  # or null for scratch
    tokenizer_path: ./tokenizer

    tasks:
      - name: BBBP
        task_type: binary_classification
        metric: roc_auc
      - name: ESOL
        task_type: regression
        metric: rmse

    data_sources:
      - path: data/bbbp.csv
        SMILES_column: SMILES
        label_mapping:
          p_np: BBBP
      - path: data/esol.csv
        SMILES_column: SMILES
        label_mapping:
          measured_log_solubility: ESOL

    training:
      output_dir: ./experiments/multitask
      num_epochs: 50
      batch_size: 32
      learning_rate: 2.0e-5
      warmup_ratio: 0.1
"""

import argparse
import logging
import os
import json
from pathlib import Path

import yaml
from transformers import PreTrainedTokenizerFast, RobertaConfig

from tasks import TaskSpec, TaskType, TaskRegistry
from config import MultiTaskConfig, DataSourceConfig
from multitask_model import MultiTaskModel
from multitask_data import (
    DataSourceSpec,
    create_dataset_from_sources,
    train_val_split,
)
from multitask_trainer import train_multitask
from utils import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_task_registry(config: dict) -> TaskRegistry:
    """Build TaskRegistry from config."""
    registry = TaskRegistry()

    for task_config in config.get('tasks', []):
        task_type = TaskType(task_config['task_type'])
        spec = TaskSpec(
            name=task_config['name'],
            task_type=task_type,
            num_classes=task_config.get('num_classes', 1),
            loss_weight=task_config.get('loss_weight', 1.0),
            metric=task_config.get('metric', ''),
            description=task_config.get('description', ''),
        )
        registry.register(spec)

    logger.info(f"Registered {len(registry)} tasks: {registry.task_names}")
    return registry


def build_data_sources(config: dict) -> list:
    """Build DataSourceSpec list from config."""
    sources = []
    for ds_config in config.get('data_sources', []):
        spec = DataSourceSpec(
            path=ds_config['path'],
            SMILES_column=ds_config.get('SMILES_column', 'SMILES'),
            label_mapping=ds_config.get('label_mapping', {}),
        )
        sources.append(spec)
    return sources


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    """Load tokenizer from directory."""
    tokenizer_file = Path(path) / "tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task supervised pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--log_file", help="Optional log file path"
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    args = parser.parse_args()

    setup_logging(args.log_file)

    logger.info("=" * 80)
    logger.info("MULTI-TASK SUPERVISED PRETRAINING")
    logger.info("=" * 80)

    # Load config
    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    # Build task registry
    task_registry = build_task_registry(config)

    # Load tokenizer
    tokenizer_path = config.get('tokenizer_path', config.get('pretrained_model_path'))
    if not tokenizer_path:
        raise ValueError("Must specify tokenizer_path or pretrained_model_path")
    tokenizer = load_tokenizer(tokenizer_path)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Build data sources
    data_sources = build_data_sources(config)
    logger.info(f"Data sources: {len(data_sources)}")

    # Create pooled dataset
    logger.info("Loading and pooling datasets...")
    max_length = config.get('model', {}).get('max_position_embeddings', 512)
    dataset = create_dataset_from_sources(
        source_specs=data_sources,
        task_registry=task_registry,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Train/val split
    if args.val_fraction > 0:
        train_dataset, val_dataset = train_val_split(
            dataset, val_fraction=args.val_fraction
        )
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Train: {len(train_dataset)} (no validation)")

    # Create model
    pretrained_path = config.get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading encoder from: {pretrained_path}")
        model = MultiTaskModel.from_pretrained_encoder(
            encoder_path=pretrained_path,
            task_registry=task_registry,
        )
    else:
        logger.info("Creating model from scratch")
        model_config = config.get('model', {})
        model_config['vocab_size'] = len(tokenizer)
        encoder_config = RobertaConfig(**model_config)
        model = MultiTaskModel.from_encoder_config(
            config=encoder_config,
            task_registry=task_registry,
        )

    # Training config
    training_config = config.get('training', {})
    output_dir = training_config.get('output_dir', './experiments/multitask')

    # Train
    results = train_multitask(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=training_config.get('num_epochs', 50),
        batch_size=training_config.get('batch_size', 32),
        learning_rate=float(training_config.get('learning_rate', 2e-5)),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        logging_steps=training_config.get('logging_steps', 100),
        save_steps=training_config.get('save_steps', 1000),
        eval_steps=training_config.get('eval_steps', 500),
        fp16=training_config.get('fp16', False),
        save_encoder=True,
    )

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {config_save_path}")

    # Save task registry
    registry_path = os.path.join(output_dir, "task_registry.json")
    with open(registry_path, 'w') as f:
        json.dump(task_registry.to_dict(), f, indent=2)

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info(f"Encoder saved to: {output_dir}/encoder")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

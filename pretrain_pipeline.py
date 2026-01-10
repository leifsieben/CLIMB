"""
pretrain_pipeline.py
====================

Orchestrate unsupervised MLM and supervised multi-task pretraining under a
fixed compute budget. The pipeline:

1. Loads the shared BPE tokenizer.
2. Builds the encoder + supervision scaffolding (task heads + optional MLM head).
3. Allocates the budget between MLM and supervised phases.
4. Runs each phase sequentially while reusing the same encoder weights.
"""

import argparse
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    RobertaConfig,
    Trainer,
    TrainingArguments,
)

from config import PretrainingConfig
from data import UnsupervisedChemicalDataset
from multitask_data import (
    DataSourceSpec,
    create_dataset_from_sources,
    train_val_split,
)
from multitask_model import MultiTaskModel
from multitask_trainer import train_multitask
from tasks import TaskRegistry, TaskSpec, TaskType
from utils import get_device, setup_logging

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load a pretrained tokenizer directory."""
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def load_tokenized_samples(path: str) -> List[Dict[str, Any]]:
    """Load a pickle containing tokenized SMILES."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        if "data" in obj:
            samples = obj["data"]
        else:
            samples = obj
    elif isinstance(obj, list):
        samples = obj
    else:
        raise ValueError(f"Unsupported tokenized format: {type(obj)}")

    if not isinstance(samples, list):
        raise ValueError(f"Tokenized samples must be a list, got {type(samples)}")
    return samples


def aggregate_unsupervised_data(paths: List[str]) -> List[Dict[str, Any]]:
    """Load and concatenate multiple tokenized unsupervised pickles."""
    data: List[Dict[str, Any]] = []
    for path in paths:
        samples = load_tokenized_samples(path)
        logger.info(f"Loaded {len(samples)} MLM samples from {path}")
        data.extend(samples)
    return data


def build_task_registry(task_configs: List[Dict[str, Any]]) -> TaskRegistry:
    """Build TaskRegistry from task configuration dictionaries."""
    registry = TaskRegistry()
    for cfg in task_configs:
        task_spec = TaskSpec(
            name=cfg["name"],
            task_type=TaskType(cfg["task_type"]),
            num_classes=int(cfg.get("num_classes", 1)),
            loss_weight=float(cfg.get("loss_weight", 1.0)),
            metric=cfg.get("metric", ""),
            description=cfg.get("description", ""),
        )
        registry.register(task_spec)
    logger.info(f"Registered {len(registry)} tasks: {registry.task_names}")
    return registry


def build_data_source_specs(configs: List[Dict[str, Any]]) -> List[DataSourceSpec]:
    """Convert raw data source dicts to DataSourceSpec instances."""
    specs = []
    for cfg in configs:
        specs.append(
            DataSourceSpec(
                path=cfg["path"],
                SMILES_column=cfg.get("SMILES_column", "SMILES"),
                label_mapping=cfg.get("label_mapping", {}),
            )
        )
    return specs


def allocate_epoch_budget(total_epochs: int, supervised_fraction: float) -> Tuple[int, int]:
    """
    Split the compute budget into MLM (unsupervised) and supervised epochs.

    Returns:
        (mlm_epochs, supervised_epochs)
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")

    supervised_fraction = max(0.0, min(supervised_fraction, 1.0))
    unsupervised_fraction = 1.0 - supervised_fraction

    if supervised_fraction == 0:
        return total_epochs, 0
    if supervised_fraction == 1:
        return 0, total_epochs

    mlm_epochs = max(1, int(round(total_epochs * unsupervised_fraction)))
    supervised_epochs = max(1, total_epochs - mlm_epochs)

    total_assigned = mlm_epochs + supervised_epochs
    if total_assigned > total_epochs:
        excess = total_assigned - total_epochs
        if unsupervised_fraction < supervised_fraction:
            supervised_epochs = max(0, supervised_epochs - excess)
        else:
            mlm_epochs = max(0, mlm_epochs - excess)

    return mlm_epochs, supervised_epochs


def train_unsupervised_phase(
    model: MultiTaskModel,
    tokenizer: PreTrainedTokenizerFast,
    dataset: UnsupervisedChemicalDataset,
    config: PretrainingConfig,
    epochs: int,
    stage_dir: Path,
) -> None:
    """Run MLM training for the allocated number of epochs."""
    mlm_cfg = config.mlm_training

    stage_dir.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_cfg.mlm_probability,
    )

    evaluation_strategy = mlm_cfg.evaluation_strategy
    eval_steps = mlm_cfg.eval_steps
    if eval_steps <= 0:
        evaluation_strategy = "no"
        eval_steps = None

    training_args = TrainingArguments(
        output_dir=str(stage_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=mlm_cfg.batch_size,
        learning_rate=mlm_cfg.learning_rate,
        weight_decay=mlm_cfg.weight_decay,
        warmup_steps=mlm_cfg.warmup_steps,
        logging_steps=mlm_cfg.logging_steps,
        save_steps=mlm_cfg.save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=mlm_cfg.save_total_limit,
        logging_dir=str(stage_dir / "logs"),
        dataloader_num_workers=mlm_cfg.dataloader_num_workers,
        gradient_accumulation_steps=mlm_cfg.gradient_accumulation_steps,
        max_grad_norm=mlm_cfg.max_grad_norm,
        fp16=mlm_cfg.fp16 and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info(f"Starting MLM pretraining for {epochs} epoch(s)")
    trainer.train()
    trainer.save_model(str(stage_dir / "final"))
    model.save_encoder(str(stage_dir / "encoder"))
    logger.info(f"MLM checkpoint saved to {stage_dir}")


def run_supervised_phase(
    model: MultiTaskModel,
    tokenizer: PreTrainedTokenizerFast,
    config: PretrainingConfig,
    task_registry: TaskRegistry,
    epochs: int,
    stage_dir: Path,
) -> Dict[str, Any]:
    """Run multi-task supervised pretraining."""
    if not config.tasks or not config.data_sources:
        raise ValueError("Supervised phase requires tasks and data_sources in the config")

    specs = build_data_source_specs(config.data_sources)
    max_length = config.model.max_position_embeddings
    dataset = create_dataset_from_sources(
        source_specs=specs,
        task_registry=task_registry,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if config.validation_fraction > 0:
        train_dataset, val_dataset = train_val_split(
            dataset,
            val_fraction=config.validation_fraction,
        )
    else:
        train_dataset = dataset
        val_dataset = None

    stage_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = config.supervised_training
    output_dir = stage_dir

    logger.info(f"Starting supervised pretraining for {epochs} epoch(s)")
    results = train_multitask(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=str(output_dir),
        num_epochs=epochs,
        batch_size=training_cfg.batch_size,
        learning_rate=training_cfg.learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps,
        fp16=training_cfg.fp16,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        max_grad_norm=training_cfg.max_grad_norm,
        dataloader_num_workers=training_cfg.dataloader_num_workers,
        save_encoder=True,
    )
    return {
        **results,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "tasks": task_registry.task_names,
    }


def save_metadata(config: PretrainingConfig, metadata: Dict[str, Any], path: Path) -> None:
    """Persist metadata for reproducibility."""
    payload = {
        "config": config.to_dict(),
        "metadata": metadata,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Pretraining pipeline for MLM + supervised mix")
    parser.add_argument("--config", required=True, help="Path to pretraining YAML config")
    parser.add_argument("--log_file", help="Optional log file")
    args = parser.parse_args()

    setup_logging(args.log_file)

    config = PretrainingConfig.from_yaml(args.config)
    root_dir = Path(config.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PRETRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {root_dir}")
    logger.info(f"Compute budget: {config.compute_budget.total_epochs} epoch(s)")
    logger.info(f"Device: {get_device()}")

    tokenizer = load_tokenizer(config.tokenizer_path)
    logger.info(f"Loaded tokenizer (vocab size={len(tokenizer)})")

    task_registry = build_task_registry(config.tasks)

    model_cfg = asdict(config.model)
    model_cfg["vocab_size"] = len(tokenizer)
    encoder_config = RobertaConfig(**model_cfg)
    model = MultiTaskModel.from_encoder_config(
        config=encoder_config,
        task_registry=task_registry,
        include_mlm_head=True,
    )

    mlm_epochs, supervised_epochs = allocate_epoch_budget(
        config.compute_budget.total_epochs,
        config.compute_budget.supervised_fraction,
    )
    logger.info(f"MLM epochs: {mlm_epochs}, supervised epochs: {supervised_epochs}")

    metadata = {
        "mlm_epochs": mlm_epochs,
        "supervised_epochs": supervised_epochs,
        "compute_budget": asdict(config.compute_budget),
        "device": get_device(),
    }

    if mlm_epochs > 0:
        if not config.unsupervised_data:
            raise ValueError("MLM phase enabled but no unsupervised_data paths were provided")

        unsup_samples = aggregate_unsupervised_data(config.unsupervised_data)
        dataset = UnsupervisedChemicalDataset(unsup_samples)
        mlm_dir = root_dir / "unsupervised"
        train_unsupervised_phase(model, tokenizer, dataset, config, mlm_epochs, mlm_dir)
        metadata["mlm_samples"] = len(dataset)
        metadata["mlm_output"] = str(mlm_dir)

    supervised_results = {}
    sup_dir = root_dir / "supervised"
    if supervised_epochs > 0:
        supervised_results = run_supervised_phase(
            model=model,
            tokenizer=tokenizer,
            config=config,
            task_registry=task_registry,
            epochs=supervised_epochs,
            stage_dir=sup_dir,
        )
        metadata.update(supervised_results)
        metadata["supervised_output"] = str(sup_dir)

    tokenizer.save_pretrained(root_dir / "tokenizer")
    config.save_yaml(str(root_dir / "config.yaml"))

    save_metadata(config, metadata, root_dir / "metadata.json")

    logger.info("PRETRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    if supervised_epochs > 0:
        logger.info(f"Final encoder: {sup_dir / 'encoder'}")
    else:
        logger.info("Supervised phase was skipped; encoder kept in MLM checkpoint")


if __name__ == "__main__":
    main()

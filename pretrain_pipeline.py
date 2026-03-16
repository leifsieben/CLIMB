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
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    RobertaConfig,
    TrainingArguments,
)

from config import PretrainingConfig
from data import (
    StreamingTokenizedDataset,
    UnsupervisedChemicalDataset,
)
from multitask_data import (
    DataSourceSpec,
    create_dataset_from_sources,
    train_val_split,
)
from multitask_model import MultiTaskModel
from multitask_trainer import train_multitask
from tasks import TaskRegistry, TaskSpec, TaskType
from utils import get_device, register_spot_handler, setup_logging
from token_budget import TokenBudgetCallback, TokenBudgetTracker, TokenBudgetTrainer
from supervised_streaming import (
    DEFAULT_FAMILIES,
    SupervisedFamily,
    build_task_registry_for_family,
    count_non_nan_labels,
    estimate_avg_tokens_from_parquet,
    parquet_has_tokenized_columns,
    resolve_family_specs,
    StreamingSupervisedFamilyDataset,
)

logger = logging.getLogger(__name__)


def _resolve_paths(paths: List[str]) -> List[str]:
    """Expand files and directories into a sorted list of pickle paths."""
    resolved = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            resolved.extend(sorted(str(x) for x in p.glob("*.pkl")))
            resolved.extend(sorted(str(x) for x in p.glob("*.parquet")))
        elif p.exists():
            resolved.append(str(p))
    if not resolved:
        raise ValueError("No tokenized files found at the specified locations")
    return resolved


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


def estimate_avg_tokens_from_tokenized_paths(paths: List[str], sample_size: int = 2000) -> float:
    """Estimate average token length from tokenized shards (pickle or parquet)."""
    seen = 0
    lengths: List[int] = []
    for path in paths:
        if path.endswith(".parquet") or Path(path).is_dir():
            try:
                import pyarrow.dataset as ds
                import pyarrow.compute as pc
            except Exception as exc:
                raise ImportError(
                    "pyarrow is required to estimate token lengths from parquet. "
                    f"Import error: {exc}"
                )
            p = Path(path)
            if p.is_dir():
                files = sorted(str(x) for x in p.glob("*.parquet"))
                if not files:
                    raise FileNotFoundError(f"No parquet files found in {path}")
                dataset = ds.dataset(files, format="parquet")
            else:
                dataset = ds.dataset(path, format="parquet")
            if "input_ids" not in dataset.schema.names:
                raise ValueError(f"Parquet missing input_ids column: {path}")
            for batch in dataset.to_batches(columns=["input_ids"], batch_size=10_000):
                lens = pc.list_value_length(batch.column(0)).to_pylist()
                for ln in lens:
                    if ln is None:
                        continue
                    lengths.append(int(ln))
                    seen += 1
                    if seen >= sample_size:
                        break
                if seen >= sample_size:
                    break
        else:
            samples = load_tokenized_samples(path)
            for item in samples:
                input_ids = item.get("input_ids")
                if input_ids is None:
                    continue
                lengths.append(len(input_ids))
                seen += 1
                if seen >= sample_size:
                    break
        if seen >= sample_size:
            break
    if not lengths:
        return 512.0
    return float(sum(lengths) / len(lengths))


def compute_max_steps(
    token_budget: int,
    batch_size: int,
    avg_tokens_per_sample: float,
    safety_factor: float = 0.8,
) -> int:
    """Compute max_steps so token budget is the limiting factor."""
    tokens_per_step = max(1, int(avg_tokens_per_sample * batch_size * safety_factor))
    return int(math.ceil(token_budget / tokens_per_step))


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
    dataset: Union[UnsupervisedChemicalDataset, StreamingTokenizedDataset],
    config: PretrainingConfig,
    epochs: int,
    stage_dir: Path,
    spot_mode: bool = False,
    token_budget: int = None,
    metrics_path: str = None,
    run_id: str = "run",
    phase: str = "unsupervised",
    max_steps: int = None,
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
        max_steps=max_steps,
    )

    token_tracker = TokenBudgetTracker(token_budget) if token_budget else None
    callbacks = []
    if token_tracker:
        callbacks.append(TokenBudgetCallback(token_tracker, metrics_path, run_id, phase))

    trainer = TokenBudgetTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        token_budget_tracker=token_tracker,
    )

    if spot_mode:
        register_spot_handler(trainer, stage_dir)

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
    spot_mode: bool = False,
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
        spot_mode=spot_mode,
    )
    return {
        **results,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "tasks": task_registry.task_names,
    }


def run_supervised_families(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    config: PretrainingConfig,
    stage_dir: Path,
    token_budget: Optional[int],
    run_id: str,
    metrics_path: str,
    spot_mode: bool = False,
) -> Dict[str, Any]:
    """Run supervised training sequentially by dataset family."""
    parquet_path = config.supervised_tokenized_parquet_path or config.supervised_parquet_path
    if not parquet_path:
        raise ValueError("supervised_parquet_path must be set for sequential family training")

    tokenized = parquet_has_tokenized_columns(parquet_path)
    smiles_col, family_specs = resolve_family_specs(
        parquet_path,
        config.supervised_families or DEFAULT_FAMILIES,
    )

    families: List[SupervisedFamily] = []
    for spec in family_specs:
        columns = spec["columns"]
        if not columns:
            logger.warning("Skipping family %s (no columns found)", spec["name"])
            continue
        label_count = count_non_nan_labels(parquet_path, columns)
        families.append(
            SupervisedFamily(
                name=spec["name"],
                prefix=spec["prefix"],
                columns=columns,
                label_count=label_count,
            )
        )

    total_labels = sum(f.label_count for f in families) or 0
    if token_budget is not None and total_labels == 0:
        raise ValueError("No supervised labels found; cannot allocate token budget")

    stage_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "families": [],
        "total_label_count": total_labels,
        "smiles_column": smiles_col,
    }

    allocated = 0
    for idx, family in enumerate(families):
        if token_budget is None:
            family_budget = None
        elif idx == len(families) - 1:
            family_budget = max(0, token_budget - allocated)
        else:
            family_budget = int(token_budget * (family.label_count / total_labels))
            allocated += family_budget

        logger.info(
            "Family %s: %d label values, token_budget=%s",
            family.name,
            family.label_count,
            family_budget if family_budget is not None else "none",
        )

        task_registry = build_task_registry_for_family(family.columns)
        family_model = MultiTaskModel(
            encoder=encoder,
            task_registry=task_registry,
            include_mlm_head=False,
        )

        dataset = StreamingSupervisedFamilyDataset(
            parquet_path=parquet_path,
            smiles_col=smiles_col,
            label_columns=family.columns,
            tokenizer=None if tokenized else tokenizer,
            max_length=config.model.max_position_embeddings,
            batch_rows=config.supervised_training.streaming_batch_rows,
            input_ids_col="input_ids" if tokenized else None,
            attention_mask_col="attention_mask" if tokenized else None,
        )

        avg_len = estimate_avg_tokens_from_parquet(
            parquet_path,
            smiles_col=smiles_col,
            tokenizer=tokenizer,
            max_length=config.model.max_position_embeddings,
            input_ids_col="input_ids" if tokenized else None,
        )
        max_steps = None
        if family_budget:
            tokens_per_step = config.supervised_training.tokens_per_step_estimate
            if tokens_per_step is not None:
                max_steps = int(math.ceil(family_budget / max(tokens_per_step, 1)))
            else:
                max_steps = compute_max_steps(
                    family_budget,
                    batch_size=config.supervised_training.batch_size,
                    avg_tokens_per_sample=avg_len,
                )

        family_dir = stage_dir / family.name
        family_dir.mkdir(parents=True, exist_ok=True)

        train_results = train_multitask(
            model=family_model,
            train_dataset=dataset,
            val_dataset=None,
            output_dir=str(family_dir),
            num_epochs=config.supervised_training.num_epochs,
            batch_size=config.supervised_training.batch_size,
            learning_rate=config.supervised_training.learning_rate,
            warmup_ratio=config.supervised_training.warmup_ratio,
            weight_decay=config.supervised_training.weight_decay,
            logging_steps=config.supervised_training.logging_steps,
            save_steps=config.supervised_training.save_steps,
            eval_steps=0,
            fp16=config.supervised_training.fp16,
            gradient_accumulation_steps=config.supervised_training.gradient_accumulation_steps,
            max_grad_norm=config.supervised_training.max_grad_norm,
            dataloader_num_workers=config.supervised_training.dataloader_num_workers,
            max_steps=max_steps,
            save_encoder=True,
            spot_mode=spot_mode,
            token_budget=family_budget,
            metrics_path=metrics_path,
            run_id=run_id,
            phase=f"supervised_{family.name}",
        )

        encoder = family_model.get_encoder()
        results["families"].append(
            {
                "name": family.name,
                "columns": len(family.columns),
                "label_count": family.label_count,
                "token_budget": family_budget,
                "avg_token_length": avg_len,
                "max_steps": max_steps,
                **train_results,
            }
        )

    return results


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
    parser.add_argument("--spot", action="store_true", help="Handle spot interruptions (save checkpoints)")
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
    if config.compute_budget.total_tokens:
        logger.info(f"Token budget: {config.compute_budget.total_tokens} tokens")
    logger.info(f"Device: {get_device()}")

    tokenizer = load_tokenizer(config.tokenizer_path)
    logger.info(f"Loaded tokenizer (vocab size={len(tokenizer)})")

    task_registry = build_task_registry(config.tasks) if config.tasks else TaskRegistry()

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

    run_id = config.name
    metrics_path = str(root_dir / "metrics.jsonl")
    token_budget_total = config.compute_budget.total_tokens
    supervised_token_budget = None
    mlm_token_budget = None
    if token_budget_total:
        supervised_token_budget = int(token_budget_total * config.compute_budget.supervised_fraction)
        mlm_token_budget = max(0, token_budget_total - supervised_token_budget)

    metadata = {
        "mlm_epochs": mlm_epochs,
        "supervised_epochs": supervised_epochs,
        "compute_budget": asdict(config.compute_budget),
        "device": get_device(),
        "run_id": run_id,
        "token_budget_total": token_budget_total,
        "token_budget_unsupervised": mlm_token_budget,
        "token_budget_supervised": supervised_token_budget,
    }

    if mlm_epochs > 0:
        if not config.unsupervised_data:
            raise ValueError("MLM phase enabled but no unsupervised_data paths were provided")

        streaming_paths = [path for path in config.unsupervised_data if Path(path).is_dir()]
        if streaming_paths:
            dataset = StreamingTokenizedDataset(
                _resolve_paths(config.unsupervised_data),
                with_labels=False,
                shuffle=config.mlm_training.shuffle,
                max_samples=config.mlm_training.streaming_max_samples,
            )
            metadata["mlm_samples"] = config.mlm_training.streaming_max_samples or "streamed"
        else:
            unsup_samples = aggregate_unsupervised_data(config.unsupervised_data)
            dataset = UnsupervisedChemicalDataset(unsup_samples)
            metadata["mlm_samples"] = len(dataset)

        max_steps = None
        if mlm_token_budget:
            avg_len = estimate_avg_tokens_from_tokenized_paths(
                _resolve_paths(config.unsupervised_data)
            )
            tokens_per_step = config.mlm_training.tokens_per_step_estimate
            if tokens_per_step is not None:
                max_steps = int(math.ceil(mlm_token_budget / max(tokens_per_step, 1)))
            else:
                max_steps = compute_max_steps(
                    mlm_token_budget,
                    batch_size=config.mlm_training.batch_size,
                    avg_tokens_per_sample=avg_len,
                )
            metadata["mlm_avg_token_length"] = avg_len
            metadata["mlm_max_steps"] = max_steps
        elif streaming_paths:
            raise ValueError("Streaming MLM requires compute_budget.total_tokens for max_steps")

        mlm_dir = root_dir / "unsupervised"
        train_unsupervised_phase(
            model,
            tokenizer,
            dataset,
            config,
            1 if max_steps else mlm_epochs,
            mlm_dir,
            spot_mode=args.spot,
            token_budget=mlm_token_budget,
            metrics_path=metrics_path,
            run_id=run_id,
            phase="unsupervised",
            max_steps=max_steps,
        )
        metadata["mlm_output"] = str(mlm_dir)

    supervised_results = {}
    sup_dir = root_dir / "supervised"
    if supervised_epochs > 0:
        if config.supervised_parquet_path or config.supervised_tokenized_parquet_path:
            supervised_results = run_supervised_families(
                encoder=model.get_encoder(),
                tokenizer=tokenizer,
                config=config,
                stage_dir=sup_dir,
                token_budget=supervised_token_budget,
                run_id=run_id,
                metrics_path=metrics_path,
                spot_mode=args.spot,
            )
        else:
            supervised_results = run_supervised_phase(
                model=model,
                tokenizer=tokenizer,
                config=config,
                task_registry=task_registry,
                epochs=supervised_epochs,
                stage_dir=sup_dir,
                spot_mode=args.spot,
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

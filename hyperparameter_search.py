"""Find optimal hyperparameters for MLM pretraining with Optuna.

Supports tokenized inputs from:
- Pickle files (`.pkl`) containing a list of records or `{"data": [...]}`
- Parquet files (`.parquet`) with token columns (e.g. `input_ids`, `attention_mask`)
"""

import argparse
import json
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from model import create_model
from utils import get_device, load_config, setup_logging

logger = logging.getLogger(__name__)


class UnsupervisedDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class OptunaPruningCallback(TrainerCallback):
    """Report eval metrics to Optuna and prune unpromising trials."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        eval_loss = float(metrics["eval_loss"])
        step = int(state.global_step)
        self.trial.report(eval_loss, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step={step}, eval_loss={eval_loss:.5f}")


class HeartbeatCallback(TrainerCallback):
    """Emit periodic heartbeat logs so long runs are visibly alive."""

    def __init__(self, trial_number: int, heartbeat_seconds: int):
        self.trial_number = trial_number
        self.heartbeat_seconds = heartbeat_seconds
        self._last_heartbeat = 0.0

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if now - self._last_heartbeat >= self.heartbeat_seconds:
            self._last_heartbeat = now
            logger.info(
                "Heartbeat | trial=%s step=%s epoch=%.4f",
                self.trial_number,
                state.global_step,
                0.0 if state.epoch is None else state.epoch,
            )


def _load_pickle_records(path: Path, max_records: Optional[int] = None) -> List[Dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        records = payload.get("data", [])
    else:
        records = payload
    if max_records is not None:
        return records[:max_records]
    return records


def _load_parquet_records(path: Path, max_records: Optional[int] = None) -> List[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to load parquet tokenized shards. "
            "Install dependencies from requirements.txt."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available_cols = set(parquet_file.schema.names)
    candidate_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    cols = [c for c in candidate_cols if c in available_cols]
    if "input_ids" not in cols:
        raise ValueError(f"{path} is missing required column 'input_ids'")

    records: List[Dict] = []
    for batch in parquet_file.iter_batches(columns=cols, batch_size=10_000):
        as_dict = batch.to_pydict()
        row_count = len(as_dict[cols[0]]) if cols else 0
        for i in range(row_count):
            row = {col: as_dict[col][i] for col in cols}
            records.append(row)
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def _load_records(path: Path, max_records: Optional[int] = None) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return _load_pickle_records(path, max_records=max_records)
    if suffix == ".parquet":
        return _load_parquet_records(path, max_records=max_records)
    raise ValueError(f"Unsupported file type for {path}. Expected .pkl or .parquet")


def _collect_input_paths(path_str: str, max_shards: Optional[int] = None) -> List[Path]:
    path = Path(path_str)
    if path.is_dir():
        pkl_paths = sorted(path.glob("*.pkl"))
        parquet_paths = sorted(path.glob("*.parquet"))
        paths = pkl_paths + parquet_paths
        if not paths:
            raise ValueError(f"No .pkl or .parquet files found in directory: {path}")
        if max_shards is not None:
            paths = paths[:max_shards]
        logger.info("Loading %d shard(s) from %s", len(paths), path)
        return paths
    return [path]


def _load_tokenized_records(
    path_str: str,
    max_samples: Optional[int] = None,
    max_shards: Optional[int] = None,
) -> List[Dict]:
    all_records: List[Dict] = []
    for path in _collect_input_paths(path_str, max_shards=max_shards):
        logger.info("Reading %s", path)
        remaining = None
        if max_samples is not None:
            remaining = max(0, max_samples - len(all_records))
            if remaining == 0:
                break
        all_records.extend(_load_records(path, max_records=remaining))
        if max_samples is not None and len(all_records) >= max_samples:
            break
    if max_samples is not None and len(all_records) > max_samples:
        all_records = all_records[:max_samples]
    return all_records


def _filter_max_len(records: List[Dict], max_len: int = 512) -> List[Dict]:
    before = len(records)
    filtered = [x for x in records if len(x.get("input_ids", [])) <= max_len]
    logger.info("Filtered long sequences: kept %s / %s", f"{len(filtered):,}", f"{before:,}")
    return filtered


def _build_pruner(args) -> optuna.pruners.BasePruner:
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=args.pruner_interval_steps,
        )
    if args.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=args.sh_min_resource,
            reduction_factor=args.sh_reduction_factor,
            min_early_stopping_rate=args.sh_min_early_stopping_rate,
        )
    if args.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=args.hb_min_resource,
            max_resource=args.hb_max_resource,
            reduction_factor=args.hb_reduction_factor,
        )
    return optuna.pruners.NopPruner()


def _write_study_snapshot(study: optuna.Study, output_path: Path, storage_url: str):
    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = study.best_params
    except ValueError:
        best_trial = None
        best_value = None
        best_params = {}

    snapshot = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_trial": None if best_trial is None else best_trial.number,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "storage": storage_url,
        "updated_at_unix": int(time.time()),
    }
    with open(output_path / "study_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)


def _make_trial_callback(output_path: Path, storage_url: str):
    def _callback(study: optuna.Study, trial: optuna.FrozenTrial):
        logger.info("Trial %s finished with state=%s value=%s", trial.number, trial.state.name, trial.value)
        _write_study_snapshot(study, output_path, storage_url)
        try:
            best_params = study.best_params
        except ValueError:
            best_params = None
        if best_params is not None:
            with open(output_path / "best_hyperparameters.json", "w") as f:
                json.dump(best_params, f, indent=2)

    return _callback


def objective(
    trial,
    tokenizer,
    train_dataset,
    eval_dataset,
    base_config,
    trial_output_root: Path,
    args,
):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    num_layers = trial.suggest_int("num_hidden_layers", 4, 12)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)

    if hidden_size == 128:
        num_heads = 4
    elif hidden_size == 256:
        num_heads = trial.suggest_categorical("num_attention_heads_256", [4, 8])
    else:
        num_heads = trial.suggest_categorical("num_attention_heads_512", [8, 16])

    logger.info(
        "Trial %s | lr=%.2e bs=%s warmup=%s hidden=%s layers=%s heads=%s wd=%.2e",
        trial.number,
        learning_rate,
        batch_size,
        warmup_steps,
        hidden_size,
        num_layers,
        num_heads,
        weight_decay,
    )

    model = create_model(
        vocab_size=len(tokenizer),
        task="mlm",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=514,
        use_flash_attention=base_config.get("model", {}).get("use_flash_attention", True),
        use_gradient_checkpointing=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    output_dir = trial_output_root / f"trial_{trial.number}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        remove_unused_columns=False,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
    )

    callbacks = [
        HeartbeatCallback(trial.number, args.heartbeat_seconds),
        OptunaPruningCallback(trial),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    try:
        trainer.train()
    except optuna.TrialPruned:
        logger.info("Trial %s pruned", trial.number)
        raise

    eval_loss = trainer.evaluate()["eval_loss"]
    logger.info("Trial %s eval_loss=%.4f", trial.number, eval_loss)

    if output_dir.exists() and not args.keep_trial_artifacts:
        shutil.rmtree(output_dir)
    return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    parser.add_argument(
        "--train_data",
        required=True,
        help="Training data file or directory (.pkl/.parquet shards supported)",
    )
    parser.add_argument(
        "--eval_data",
        help="Eval data file or directory (optional; if omitted, train split is used)",
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for this worker")

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on loaded train samples to keep memory bounded",
    )
    parser.add_argument(
        "--max_train_shards",
        type=int,
        default=None,
        help="Optional cap on number of train shards loaded (directory input only)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on loaded eval samples to keep memory bounded",
    )
    parser.add_argument(
        "--max_eval_shards",
        type=int,
        default=None,
        help="Optional cap on number of eval shards loaded (directory input only)",
    )

    parser.add_argument("--log_file", help="Log file path")
    parser.add_argument("--study_name", default="mlm_hpo", help="Optuna study name")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Default uses sqlite in output directory.",
    )
    parser.add_argument(
        "--load_if_exists",
        action="store_true",
        help="Resume existing study if present",
    )
    parser.add_argument(
        "--no_load_if_exists",
        action="store_false",
        dest="load_if_exists",
        help="Do not resume even if study exists",
    )
    parser.set_defaults(load_if_exists=True)
    parser.add_argument(
        "--worker_id",
        default="worker-0",
        help="Worker id for logs when multiple workers share one study",
    )

    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--heartbeat_seconds", type=int, default=120)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--no_dataloader_pin_memory", action="store_false", dest="dataloader_pin_memory")
    parser.set_defaults(dataloader_pin_memory=True)

    parser.add_argument("--keep_trial_artifacts", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision")

    parser.add_argument(
        "--pruner",
        choices=["none", "median", "successive_halving", "hyperband"],
        default="hyperband",
    )
    parser.add_argument("--pruner_startup_trials", type=int, default=8)
    parser.add_argument("--pruner_warmup_steps", type=int, default=100)
    parser.add_argument("--pruner_interval_steps", type=int, default=50)
    parser.add_argument("--sh_min_resource", type=int, default=100)
    parser.add_argument("--sh_reduction_factor", type=int, default=3)
    parser.add_argument("--sh_min_early_stopping_rate", type=int, default=0)
    parser.add_argument("--hb_min_resource", type=int, default=100)
    parser.add_argument("--hb_max_resource", type=int, default=5000)
    parser.add_argument("--hb_reduction_factor", type=int, default=3)

    args = parser.parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one precision mode: --bf16 or --fp16")

    setup_logging(args.log_file)
    logger.info("Starting hyperparameter search | worker=%s", args.worker_id)
    logger.info("Trials requested for this worker: %d", args.n_trials)
    logger.info("Device: %s", get_device())

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trial_output_root = output_path / "_trial_tmp"
    trial_output_root.mkdir(parents=True, exist_ok=True)

    storage_url = args.storage or f"sqlite:///{(output_path / 'optuna_study.db').resolve()}"
    logger.info("Optuna storage: %s", storage_url)

    base_config = load_config(args.config)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    train_records = _filter_max_len(
        _load_tokenized_records(
            args.train_data,
            max_samples=args.max_samples,
            max_shards=args.max_train_shards,
        )
    )
    full_dataset = UnsupervisedDataset(train_records)

    if args.eval_data:
        eval_records = _filter_max_len(
            _load_tokenized_records(
                args.eval_data,
                max_samples=args.max_eval_samples,
                max_shards=args.max_eval_shards,
            )
        )
        eval_dataset = UnsupervisedDataset(eval_records)
        train_dataset = full_dataset
    else:
        train_size = int(0.9 * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

    logger.info("Train samples: %s", f"{len(train_dataset):,}")
    logger.info("Eval samples: %s", f"{len(eval_dataset):,}")

    pruner = _build_pruner(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=args.load_if_exists,
        direction="minimize",
        pruner=pruner,
    )

    trial_callback = _make_trial_callback(output_path, storage_url)
    _write_study_snapshot(study, output_path, storage_url)

    try:
        study.optimize(
            lambda trial: objective(
                trial,
                tokenizer,
                train_dataset,
                eval_dataset,
                base_config,
                trial_output_root,
                args,
            ),
            n_trials=args.n_trials,
            callbacks=[trial_callback],
            gc_after_trial=True,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; study state remains in persistent storage.")

    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = study.best_params
    except ValueError:
        best_trial = None
        best_value = None
        best_params = {}

    if best_trial is not None:
        with open(output_path / "best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=2)

    with open(output_path / "study_summary.json", "w") as f:
        json.dump(
            {
                "study_name": study.study_name,
                "best_trial": None if best_trial is None else best_trial.number,
                "best_value": best_value,
                "best_params": best_params,
                "trials": len(study.trials),
                "storage": storage_url,
            },
            f,
            indent=2,
        )

    if trial_output_root.exists() and not args.keep_trial_artifacts:
        shutil.rmtree(trial_output_root)

    if best_trial is None:
        logger.info("No completed trial yet. Check study storage for pruned/failed trial details.")
    else:
        logger.info("Best trial: %s", best_trial.number)
        logger.info("Best eval loss: %.4f", best_value)
        logger.info("Best hyperparameters: %s", best_params)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

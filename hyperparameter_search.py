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
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
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


def objective(trial, tokenizer, train_dataset, eval_dataset, base_config, trial_output_root: Path):
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
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=10000,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    eval_loss = trainer.evaluate()["eval_loss"]
    logger.info("Trial %s eval_loss=%.4f", trial.number, eval_loss)

    if output_dir.exists():
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
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
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
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger.info("Starting hyperparameter search")
    logger.info("Trials: %d", args.n_trials)
    logger.info("Device: %s", get_device())

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

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trial_output_root = output_path / "_trial_tmp"
    trial_output_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, tokenizer, train_dataset, eval_dataset, base_config, trial_output_root
        ),
        n_trials=args.n_trials,
    )

    best_params_file = output_path / "best_hyperparameters.json"
    with open(best_params_file, "w") as f:
        json.dump(study.best_params, f, indent=2)

    study_file = output_path / "optuna_study.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    if trial_output_root.exists():
        shutil.rmtree(trial_output_root)

    logger.info("Best trial: %s", study.best_trial.number)
    logger.info("Best eval loss: %.4f", study.best_value)
    logger.info("Best hyperparameters: %s", study.best_params)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

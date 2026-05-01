"""v2 single mixed-batch pretraining loop.

Replaces v1's pretrain_pipeline.py. Removes the phase-transition state machine
entirely: there is one optimizer loop. Each batch is independently MLM-mode or
supervised-mode at the configured ratio. The model has both an MLM head
(RobertaLMHead) and a supervised multi-task head; only the active head is used
per batch.

Critical features:
- Tracks `forward_passes_seen` precisely (one FP per example per batch).
- Writes `metrics.jsonl` every `log_every_steps` and a fresh `heartbeat.json` every
  log step (atomic write via os.replace).
- Cosine LR schedule annealing over the actual `total_forward_passes`.
- Saves only the encoder at end of training (heads are discarded).

Usage:
    python pretrain_v2.py --run_dir <out_dir> --config <config.yaml>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from config_v2 import (
    SUPERVISED_FAMILIES_V2,
    EvalConfigV2,
    ModelConfigV2,
    TrainingConfigV2,
)
from data_v2 import (
    MLMCollator,
    MixedModeBatchIterator,
    SupervisedCollator,
    SupervisedInRAMDataset,
    load_supervised_inram,
    make_mlm_dataset,
)
from storage_utils import materialize_tokenizer_dir


# ---------- multi-task supervised head ----------

class SupervisedMultiTaskHead(nn.Module):
    def __init__(self, hidden_size: int, n_tasks: int, task_types: List[str]):
        super().__init__()
        self.linear = nn.Linear(hidden_size, n_tasks)
        # Per-column task type for loss routing
        self.register_buffer(
            "is_classification",
            torch.tensor([1 if t == "classification" else 0 for t in task_types], dtype=torch.bool),
        )

    def forward(self, hidden_cls: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_cls)

    def loss(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """NaN-masked multi-task loss. preds and labels: [B, T].
        Computed in float32 regardless of input dtype, for numerical stability under
        bf16 autocast.
        """
        preds_f32 = preds.float()
        labels_f32 = labels.float()
        valid = torch.isfinite(labels_f32)
        if valid.sum() == 0:
            return preds_f32.sum() * 0.0

        cls_cols = self.is_classification.unsqueeze(0).expand_as(labels_f32)

        cls_valid = valid & cls_cols
        cls_loss = torch.zeros((), device=preds.device, dtype=torch.float32)
        if cls_valid.any():
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                preds_f32[cls_valid], labels_f32[cls_valid], reduction="mean"
            )

        reg_valid = valid & ~cls_cols
        reg_loss = torch.zeros((), device=preds.device, dtype=torch.float32)
        if reg_valid.any():
            reg_loss = nn.functional.mse_loss(
                preds_f32[reg_valid], labels_f32[reg_valid], reduction="mean"
            )

        n_modes = (1 if cls_valid.any() else 0) + (1 if reg_valid.any() else 0)
        if n_modes == 0:
            return preds_f32.sum() * 0.0
        return (cls_loss + reg_loss) / max(n_modes, 1)


# ---------- v2 wrapper model ----------

class ClimbV2Model(nn.Module):
    """RobertaModel + MLM head + supervised multi-task head."""

    def __init__(
        self,
        roberta_config: RobertaConfig,
        n_supervised_tasks: int,
        supervised_task_types: List[str],
    ):
        super().__init__()
        self.config = roberta_config
        self.encoder = RobertaModel(roberta_config, add_pooling_layer=False)
        self.mlm_head = RobertaLMHead(roberta_config)
        # Supervised head only created if we have any supervised tasks
        self.has_sup = n_supervised_tasks > 0
        if self.has_sup:
            self.sup_head = SupervisedMultiTaskHead(
                roberta_config.hidden_size, n_supervised_tasks, supervised_task_types
            )

    def forward_mlm(self, input_ids, attention_mask, labels):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)
        loss = nn.functional.cross_entropy(
            prediction_scores.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def forward_sup(self, input_ids, attention_mask, labels):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = out.last_hidden_state[:, 0, :]  # [B, H]
        preds = self.sup_head(cls_hidden)
        return self.sup_head.loss(preds, labels)

    def save_encoder(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(path)


# ---------- atomic file write ----------

def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".heartbeat_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


# ---------- main training loop ----------

def train(args) -> int:
    cfg_path = Path(args.config)
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    heartbeat_path = run_dir / "heartbeat.json"
    encoder_save_path = run_dir / "encoder"
    metadata_path = run_dir / "metadata.json"

    # Resolve sub-configs
    model_cfg = ModelConfigV2(**cfg.get("model", {}))
    train_cfg = TrainingConfigV2(**cfg.get("training", {}))

    selection = cfg.get("selection", {})
    mixing_ratio = float(selection.get("mixing_ratio", 1.0))
    n_families = int(selection.get("n_families", 0))
    family_order = selection.get("family_order") or SUPERVISED_FAMILIES_V2[:n_families]
    pretraining_seed = int(selection.get("pretraining_seed", train_cfg.seed))
    total_fps = int(selection.get("total_forward_passes", train_cfg.total_forward_passes))

    set_seed(pretraining_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer ----
    tokenizer_path = cfg["tokenizer_path"]
    tokenizer_local = materialize_tokenizer_dir(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_local)
    vocab_size = tokenizer.vocab_size or model_cfg.vocab_size
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    special_tokens = [
        tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.bos_token_id,
        tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id,
    ]
    special_tokens = [t for t in special_tokens if t is not None]

    # ---- supervised in-RAM data ----
    sup_dataset = None
    sup_loader = None
    sup_task_types: List[str] = []
    if n_families > 0:
        sup_parquet = cfg["supervised_tokenized_parquet_path"]
        max_sup_rows = cfg.get("max_supervised_rows")  # None for full
        # 256 tokens covers >99% of SMILES; halves the in-RAM input memory.
        sup_max_length = min(256, model_cfg.max_position_embeddings - 2)
        ids, mask, labels, label_cols, task_types = load_supervised_inram(
            sup_parquet, family_order, max_length=sup_max_length,
            max_rows=max_sup_rows,
        )
        sup_dataset = SupervisedInRAMDataset(ids, mask, labels, task_types)
        sup_task_types = task_types
        sup_loader = DataLoader(
            sup_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=0,  # in-RAM; no need for workers
            collate_fn=SupervisedCollator(pad_token_id=pad_token_id),
            drop_last=True,
        )
        print(f"[pretrain_v2] supervised: {len(sup_dataset)} rows × {len(label_cols)} tasks")

    # ---- MLM streaming data ----
    mlm_loader = None
    if mixing_ratio > 0.0:
        mlm_paths = cfg["unsupervised_data_paths"]
        if isinstance(mlm_paths, str):
            mlm_paths = [mlm_paths]
        mlm_subset_fraction_raw = cfg.get("unsupervised_subset_fraction")
        mlm_subset_fraction = float(mlm_subset_fraction_raw) if mlm_subset_fraction_raw is not None else None
        mlm_dataset = make_mlm_dataset(
            mlm_paths,
            subset_fraction=mlm_subset_fraction,
            subset_seed=pretraining_seed,
        )
        mlm_collator = MLMCollator(
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            mlm_probability=train_cfg.mlm_probability,
            pad_token_id=pad_token_id,
            special_tokens=special_tokens,
        )
        mlm_loader = DataLoader(
            mlm_dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.dataloader_num_workers,
            collate_fn=mlm_collator,
        )

    # ---- model ----
    n_supervised_tasks = sup_dataset.labels.shape[1] if sup_dataset is not None else 0
    roberta_config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
        max_position_embeddings=model_cfg.max_position_embeddings,
        type_vocab_size=model_cfg.type_vocab_size,
        layer_norm_eps=model_cfg.layer_norm_eps,
        hidden_dropout_prob=model_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
        pad_token_id=pad_token_id,
    )
    model = ClimbV2Model(roberta_config, n_supervised_tasks, sup_task_types).to(device)

    # ---- optimizer + schedule ----
    total_steps = max(1, total_fps // train_cfg.batch_size)
    warmup_steps = max(1, int(train_cfg.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- mixed iterator ----
    iterator = MixedModeBatchIterator(
        mlm_loader=mlm_loader,
        sup_loader=sup_loader,
        mixing_ratio=mixing_ratio,
        total_batches=total_steps,
        seed=pretraining_seed,
    )

    # ---- metadata ----
    metadata = {
        "run_id": cfg.get("run_id", run_dir.name),
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "selection": selection,
        "n_supervised_tasks": n_supervised_tasks,
        "total_steps": total_steps,
        "total_forward_passes_target": total_fps,
        "warmup_steps": warmup_steps,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # ---- training loop ----
    use_amp = train_cfg.bf16 and torch.cuda.is_available()
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if use_amp else {"enabled": False}

    model.train()
    forward_passes_seen = 0
    start_time = time.time()
    last_log_time = start_time
    metrics_file = metrics_path.open("a", buffering=1)

    def heartbeat(status: str):
        _atomic_write_json(heartbeat_path, {
            "last_check_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics_mtime_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(metrics_path.stat().st_mtime if metrics_path.exists() else time.time()),
            ),
            "phase": "training",
            "status": status,
            "pid": os.getpid(),
            "forward_passes_seen": forward_passes_seen,
            "total_forward_passes_target": total_fps,
        })

    heartbeat("starting")

    try:
        for step, (mode, batch) in enumerate(iterator):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(**autocast_kwargs) if use_amp else _NoCtx():
                if mode == "mlm":
                    loss = model.forward_mlm(
                        batch["input_ids"], batch["attention_mask"], batch["labels"]
                    )
                else:
                    loss = model.forward_sup(
                        batch["input_ids"], batch["attention_mask"], batch["labels"]
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            forward_passes_seen += batch["input_ids"].shape[0]

            if (step + 1) % train_cfg.log_every_steps == 0 or step == total_steps - 1:
                now = time.time()
                rec = {
                    "step": step + 1,
                    "mode": mode,
                    "loss": float(loss.detach().item()),
                    "lr": scheduler.get_last_lr()[0],
                    "forward_passes_seen": forward_passes_seen,
                    "elapsed_seconds": now - start_time,
                    "step_time_seconds": (now - last_log_time) / train_cfg.log_every_steps,
                    "timestamp": now,
                }
                metrics_file.write(json.dumps(rec) + "\n")
                metrics_file.flush()
                os.fsync(metrics_file.fileno())
                last_log_time = now
                heartbeat("ok")
                print(f"[step {step+1}/{total_steps}] mode={mode} loss={rec['loss']:.4f} fp={forward_passes_seen}/{total_fps}", flush=True)

        heartbeat("saving")
        model.save_encoder(str(encoder_save_path))
        # also copy the tokenizer next to the encoder so eval can load it self-contained
        tokenizer_dest = run_dir / "tokenizer"
        tokenizer_dest.mkdir(exist_ok=True)
        for fn in os.listdir(tokenizer_local):
            src = Path(tokenizer_local) / fn
            if src.is_file():
                shutil.copy2(src, tokenizer_dest / fn)
        heartbeat("done")
        return 0
    except Exception as exc:
        print(f"[pretrain_v2] FAILED: {exc}", file=sys.stderr)
        heartbeat("failed")
        raise
    finally:
        metrics_file.close()


class _NoCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    sys.exit(train(args))


if __name__ == "__main__":
    main()

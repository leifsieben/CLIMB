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
import hashlib
import json
import math
import os

# One BLAS thread per process, set BEFORE numpy loads OpenBLAS -- the dataloader workers fork from
# here and inherit it. Without this the MTR collate is 8x slower than it should be on a live-
# descriptor run: rdkit's Ipc descriptor builds a characteristic polynomial through numpy.dot, so
# every worker opens its own OpenBLAS pool, six workers ask for ~20 cores on 16, and two thirds of
# the machine goes to kernel time fighting over them. Runs reading PRECOMPUTED descriptors never
# call rdkit in the worker and never noticed. Setting it here rather than in the launcher's
# environment is deliberate: an export in a shell wrapper is one indirection away from the process
# that needs it, and it was silently not reaching this one.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from config_v2 import (
    DESCRIPTOR_STATS_PATH,
    SUPERVISED_FAMILIES_V2,
    SUPERVISED_FAMILY_CAPS,
    SUPERVISED_FAMILY_WEIGHTS,
    EvalConfigV2,
    ModelConfigV2,
    TrainingConfigV2,
    build_modernbert_config,
)
from data_v2 import (
    CorruptedCollator,
    MLMCollator,
    MTRCollator,
    MultiObjectiveBatchIterator,
    RawSmilesMLMCollator,
    SupervisedCollator,
    SupervisedInRAMDataset,
    load_supervised_inram_stratified,
    make_mlm_dataset,
    make_raw_smiles_dataset,
)
from featurize_v2 import pool
from storage_utils import materialize_tokenizer_dir


# ---------- supervised multi-head (MiniMol-style) + dense MTR head ----------

class MultiHeadSupervised(nn.Module):
    """Per-family 2-layer MLP heads on the shared pooled embedding (MiniMol), with
    per-family NaN-masked loss (BCE for classification, MAE/MSE for regression) and
    either fixed per-family weights or learned homoscedastic-uncertainty weighting
    (Kendall et al.) so noisy/near-unlearnable families can't dominate the gradient.

    ``.loss(pooled, labels)`` returns ``(total, {family: float})`` for logging.
    """

    def __init__(self, hidden: int, col_family: List[str], task_types: List[str],
                 family_weights: Dict[str, float], regression_loss: str = "mae",
                 uncertainty: bool = True):
        super().__init__()
        self.families = list(dict.fromkeys(col_family))  # unique, order-preserving
        self.regression_loss = regression_loss
        self.uncertainty = uncertainty
        self.heads = nn.ModuleDict()
        for f in self.families:
            idx = [i for i, cf in enumerate(col_family) if cf == f]
            self.heads[f] = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, len(idx))
            )
            self.register_buffer(f"cols_{f}", torch.tensor(idx, dtype=torch.long))
            self.register_buffer(
                f"isclf_{f}",
                torch.tensor([task_types[i] == "classification" for i in idx], dtype=torch.bool),
            )
        if uncertainty:
            self.log_var = nn.ParameterDict({f: nn.Parameter(torch.zeros(())) for f in self.families})
        self.family_weights = {f: float(family_weights.get(f, 1.0)) for f in self.families}

    def loss(self, pooled: torch.Tensor, labels: torch.Tensor):
        labels = labels.float()
        per_family: Dict[str, float] = {}
        terms = []
        for f in self.families:
            cols = getattr(self, f"cols_{f}")
            isclf = getattr(self, f"isclf_{f}")
            preds = self.heads[f](pooled).float()          # [B, nf]
            y = labels.index_select(1, cols)
            valid = torch.isfinite(y)
            if valid.sum() == 0:
                Lf = preds.sum() * 0.0
            else:
                clf_m = valid & isclf.unsqueeze(0)
                reg_m = valid & (~isclf).unsqueeze(0)
                parts = []
                if clf_m.any():
                    parts.append(nn.functional.binary_cross_entropy_with_logits(preds[clf_m], y[clf_m]))
                if reg_m.any():
                    parts.append(nn.functional.l1_loss(preds[reg_m], y[reg_m])
                                 if self.regression_loss == "mae"
                                 else nn.functional.mse_loss(preds[reg_m], y[reg_m]))
                Lf = sum(parts) / max(len(parts), 1) if parts else preds.sum() * 0.0
            per_family[f] = float(Lf.detach())
            if self.uncertainty:
                s = self.log_var[f]
                terms.append(torch.exp(-s) * Lf + s)       # Kendall: exp(-log σ²)·L + log σ²
            else:
                terms.append(self.family_weights[f] * Lf)
        if self.uncertainty:
            total = sum(terms)
        else:
            wsum = sum(self.family_weights.values()) or 1.0
            total = sum(terms) / wsum
        return total, per_family


class MTRHead(nn.Module):
    """Dense descriptor regression head (ChemBERTa-2 MTR): pooled → ~200 descriptors."""

    def __init__(self, hidden: int, n_desc: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_desc))

    def loss(self, pooled: torch.Tensor, targets: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        preds = self.net(pooled).float()
        t = targets.float()
        valid = torch.isfinite(t)
        if valid.sum() == 0:
            return preds.sum() * 0.0
        return (nn.functional.l1_loss(preds[valid], t[valid]) if loss_type == "mae"
                else nn.functional.mse_loss(preds[valid], t[valid]))


# ---------- v2 wrapper model ----------

class ClimbV2Model(nn.Module):
    """ModernBERT encoder + optional {MLM, supervised multi-head, dense MTR} objectives.

    All objectives share the encoder and the SAME masked-mean pooling used at eval.
    Only the base encoder is saved; all heads are discarded. Supports warm-starting the
    encoder from a prior run (sequential unsup→sup).
    """

    def __init__(self, modernbert_config, *, supervised: Optional[dict] = None,
                 mtr_n_desc: int = 0):
        super().__init__()
        self.config = modernbert_config
        self.mlm_model = ModernBertForMaskedLM(modernbert_config)
        self.encoder = self.mlm_model.model  # ModernBertModel (shared weights)
        h = modernbert_config.hidden_size
        self.sup_head = MultiHeadSupervised(h, **supervised) if supervised else None
        self.mtr_head = MTRHead(h, mtr_n_desc) if mtr_n_desc > 0 else None

    def _pool(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return pool(out.last_hidden_state, attention_mask, mode="mean")

    def forward_mlm(self, input_ids, attention_mask, labels):
        return self.mlm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    def forward_sup(self, input_ids, attention_mask, labels):
        return self.sup_head.loss(self._pool(input_ids, attention_mask), labels)  # (total, per_family)

    def forward_mtr(self, input_ids, attention_mask, targets, loss_type="mse"):
        return self.mtr_head.loss(self._pool(input_ids, attention_mask), targets, loss_type)

    def load_init_encoder(self, path: str):
        from transformers import ModernBertModel
        # Every warm-start path in this project is a LOCAL directory. If it is missing,
        # from_pretrained falls back to treating it as a Hugging Face repo id and raises
        # HFValidationError about repo-id syntax -- which reads like a bad config string and sends
        # you looking in the manifest, when the actual fault is that nobody staged the weights.
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"warm-start encoder directory {path!r} does not exist on this box. It is a local "
                f"path, not a Hub id -- stage it (aws s3 sync s3://climb-s3-bucket/{path} {path}) "
                f"before launching.")
        src = ModernBertModel.from_pretrained(path, reference_compile=False)
        self.encoder.load_state_dict(src.state_dict())

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
    pretraining_seed = int(selection.get("pretraining_seed", train_cfg.seed))
    total_fps = int(selection.get("total_forward_passes", train_cfg.total_forward_passes))
    augmentation = selection.get("augmentation", train_cfg.augmentation)
    _subset_raw = selection.get("unsupervised_subset_fraction", cfg.get("unsupervised_subset_fraction"))
    mlm_subset_fraction = float(_subset_raw) if _subset_raw is not None else None
    init_encoder_path = selection.get("init_encoder_path")
    # E13 / H2c content-free control: same objective + compute, zero real chemistry.
    # None (default) | "shuffle_tokens" (MLM) | "shuffle_targets" (MTR) | "both".
    corruption = selection.get("corruption")

    # Per-batch objective weights. Explicit `objectives` supersedes the legacy
    # mixing_ratio/n_families knobs (kept for backward compatibility).
    objectives = selection.get("objectives")
    if not objectives:
        mixing_ratio = float(selection.get("mixing_ratio", 1.0))
        objectives = {}
        if mixing_ratio > 0.0:
            objectives["mlm"] = mixing_ratio
        if mixing_ratio < 1.0:
            objectives["supervised"] = 1.0 - mixing_ratio
    objectives = {k: float(v) for k, v in objectives.items() if v and float(v) > 0}

    sup_families = selection.get("supervised_families")
    if sup_families is None and "supervised" in objectives:
        n_families = int(selection.get("n_families", 0))
        sup_families = selection.get("family_order") or (SUPERVISED_FAMILIES_V2[:n_families] if n_families else SUPERVISED_FAMILIES_V2)

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

    seq_cap = min(train_cfg.train_max_length, model_cfg.max_position_embeddings - 2)
    raw_paths = cfg.get("unsupervised_raw_smiles_paths")
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]

    # ---- supervised (stratified, family-balanced) ----
    supervised_spec = None
    sup_loader = None
    if "supervised" in objectives:
        caps = {f: SUPERVISED_FAMILY_CAPS.get(f, 100_000) for f in sup_families}
        # eval-molecule blocklist (leakage dedup); local path or s3:// (downloaded once)
        blocklist = None
        bl_path = cfg.get("eval_blocklist_path")
        if bl_path:
            if str(bl_path).startswith("s3://"):
                import subprocess, tempfile
                local_bl = str(Path(tempfile.gettempdir()) / "eval_blocklist.json")
                subprocess.run(["aws", "s3", "cp", bl_path, local_bl], check=True)
                bl_path = local_bl
            blocklist = set(json.loads(Path(bl_path).read_text()))
            print(f"[pretrain_v2] eval blocklist: {len(blocklist)} molecules will be dropped from SFT", flush=True)
        ids, mask, labels, label_cols, task_types, col_family = load_supervised_inram_stratified(
            cfg["supervised_tokenized_parquet_path"], sup_families, caps,
            max_length=seq_cap, regression_loss=train_cfg.supervised_regression_loss,
            blocklist=blocklist,
        )
        sup_dataset = SupervisedInRAMDataset(ids, mask, labels, task_types)
        sup_loader = DataLoader(
            sup_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0,
            collate_fn=SupervisedCollator(pad_token_id=pad_token_id), drop_last=True,
        )
        supervised_spec = dict(
            col_family=col_family, task_types=task_types,
            family_weights=SUPERVISED_FAMILY_WEIGHTS,
            regression_loss=train_cfg.supervised_regression_loss,
            uncertainty=train_cfg.uncertainty_weighting,
        )
        print(f"[pretrain_v2] supervised: {len(sup_dataset)} rows × {len(label_cols)} tasks "
              f"over {sorted(set(col_family))}")

    # ---- dense descriptor MTR ----
    mtr_loader = None
    mtr_n_desc = 0
    mtr_stats_sha1 = None
    if "mtr" in objectives:
        from descriptors_v2 import fit_descriptor_stats, load_stats, save_stats
        if not raw_paths:
            raise ValueError("mtr objective requires 'unsupervised_raw_smiles_paths'")
        stats_path = cfg.get("descriptor_stats_path", DESCRIPTOR_STATS_PATH)
        # The dangerous branch announced itself and the safe one was silent, so "did this run refit
        # the normalizer or load the canonical one?" could only be answered by inference from an
        # unchanged mtime. A refit normalizer is a different loss surface: a run that refits is not
        # a control for a run that did not, and nothing anywhere errors. Say which, and record the
        # digest in metadata.json so the answer outlives the log.
        if Path(stats_path).exists():
            desc_stats = load_stats(stats_path)
            mtr_stats_sha1 = hashlib.sha1(Path(stats_path).read_bytes()).hexdigest()[:12]
            print(f"[pretrain_v2] MTR stats LOADED from {stats_path} "
                  f"({len(desc_stats['mean'])} descriptors, sha1:{mtr_stats_sha1}) -- not refit",
                  flush=True)
        else:
            print("[pretrain_v2] fitting descriptor stats on a raw-SMILES sample ...", flush=True)
            sample = [ex["smiles"] for _, ex in zip(range(20000), make_raw_smiles_dataset(raw_paths, subset_seed=0))]
            desc_stats = fit_descriptor_stats(sample)
            save_stats(desc_stats, stats_path)
            mtr_stats_sha1 = hashlib.sha1(Path(stats_path).read_bytes()).hexdigest()[:12]
            print(f"[pretrain_v2] MTR stats REFIT on this corpus (sha1:{mtr_stats_sha1}) -- this "
                  f"run is NOT comparable to runs normalized with different stats", flush=True)
        mtr_n_desc = len(desc_stats["mean"])
        desc_dir = cfg.get("descriptor_precompute_dir")
        if desc_dir:
            print(f"[pretrain_v2] MTR using PRECOMPUTED descriptors from {desc_dir} (GPU-bound)", flush=True)
        mtr_collator = MTRCollator(tokenizer, desc_stats, max_length=seq_cap,
                                   randomize=(augmentation == "enumerated"))
        if corruption in ("shuffle_targets", "both"):
            mtr_collator = CorruptedCollator(mtr_collator, "shuffle_targets", seed=pretraining_seed)
            print("[pretrain_v2] E13 CONTROL: MTR targets SHUFFLED across batch "
                  "(molecule→descriptor mapping destroyed)", flush=True)
        mtr_loader = DataLoader(
            make_raw_smiles_dataset(raw_paths, subset_fraction=mlm_subset_fraction,
                                    subset_seed=pretraining_seed, descriptors_dir=desc_dir),
            batch_size=train_cfg.batch_size, num_workers=train_cfg.dataloader_num_workers,
            collate_fn=mtr_collator,
        )

    # ---- MLM streaming ----
    mlm_loader = None
    if "mlm" in objectives:
        mlm_max_length = min(train_cfg.train_max_length, model_cfg.max_position_embeddings)
        # "canonical_raw": canonical SMILES tokenized ON THE FLY with this run's tokenizer,
        # instead of reading the vocab-1000-baked pre-tokenized cache. Needed for the vocab-size
        # sweep, where every run has a different tokenizer/vocab: the cache cannot be used, and we
        # do NOT want enumeration (that would confound vocab with augmentation), so randomize=False.
        if augmentation in ("enumerated", "canonical_raw"):
            if not raw_paths:
                raise ValueError(f"augmentation='{augmentation}' requires 'unsupervised_raw_smiles_paths'")
            mlm_dataset = make_raw_smiles_dataset(raw_paths, subset_fraction=mlm_subset_fraction, subset_seed=pretraining_seed)
            mlm_collator = RawSmilesMLMCollator(
                tokenizer, mlm_probability=train_cfg.mlm_probability,
                randomize=(augmentation == "enumerated"),
                max_length=mlm_max_length, special_tokens=special_tokens,
            )
        else:
            mlm_paths = cfg["unsupervised_data_paths"]
            if isinstance(mlm_paths, str):
                mlm_paths = [mlm_paths]
            mlm_dataset = make_mlm_dataset(mlm_paths, subset_fraction=mlm_subset_fraction, subset_seed=pretraining_seed)
            mlm_collator = MLMCollator(
                mask_token_id=mask_token_id, vocab_size=vocab_size,
                mlm_probability=train_cfg.mlm_probability, pad_token_id=pad_token_id,
                special_tokens=special_tokens, max_length=mlm_max_length,
            )
        if corruption in ("shuffle_tokens", "both"):
            mlm_collator = CorruptedCollator(mlm_collator, "shuffle_tokens", seed=pretraining_seed)
            print("[pretrain_v2] E13 CONTROL: MLM token order SHUFFLED within each sequence "
                  "(SMILES grammar destroyed; token distribution + mask rate preserved)", flush=True)
        mlm_loader = DataLoader(mlm_dataset, batch_size=train_cfg.batch_size,
                                num_workers=train_cfg.dataloader_num_workers, collate_fn=mlm_collator)

    # ---- model ----
    attn_impl = "sdpa"
    modernbert_config = build_modernbert_config(
        model_cfg, vocab_size, pad_token_id=pad_token_id,
        bos_token_id=(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0),
        eos_token_id=(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2),
        cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id,
        attn_implementation=attn_impl,
    )
    model = ClimbV2Model(modernbert_config, supervised=supervised_spec, mtr_n_desc=mtr_n_desc)
    if init_encoder_path:
        print(f"[pretrain_v2] warm-starting encoder from {init_encoder_path}", flush=True)
        model.load_init_encoder(init_encoder_path)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"[pretrain_v2] ModernBERT {n_params/1e6:.1f}M | objectives={objectives} | aug={augmentation}"
          f"{' | init='+init_encoder_path if init_encoder_path else ''}"
          f"{' | CORRUPTION='+corruption if corruption else ''}", flush=True)

    # ---- optimizer + schedule ----
    total_steps = max(1, total_fps // train_cfg.batch_size)
    warmup_steps = max(1, int(train_cfg.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    iterator = MultiObjectiveBatchIterator(
        {"mlm": mlm_loader, "mtr": mtr_loader, "supervised": sup_loader},
        objectives, total_batches=total_steps, seed=pretraining_seed,
    )

    # ---- what machine this ran on -------------------------------------------------------------
    # Reconstructing a run's throughput took three steps because no artefact recorded the hardware:
    # skip_dense_48M hit 762 seq/s on live descriptors and a 16-vCPU box gets 302, and the split
    # between "fewer descriptors" and "wider box" could only be inferred. A rate is not a property
    # of the code alone. Best effort and never fatal -- IMDS is a 2-second timeout away and a run
    # must not die because a metadata endpoint was slow.
    def _host_facts():
        facts = {"nproc": os.cpu_count()}
        try:
            import urllib.request
            base = "http://169.254.169.254/latest/meta-data/"
            for field, key in (("instance-type", "instance_type"), ("instance-id", "instance_id")):
                with urllib.request.urlopen(base + field, timeout=2) as r:
                    facts[key] = r.read().decode()
        except Exception:
            pass
        try:
            facts["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        try:
            import rdkit
            facts["rdkit"] = rdkit.__version__
        except Exception:
            pass
        return facts

    # ---- metadata ----
    metadata = {
        "run_id": cfg.get("run_id", run_dir.name),
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "selection": selection,
        "objectives": objectives,
        "corruption": corruption,          # E13/H2c: content-free control marker (None = real pretraining)
        "supervised_families": sup_families,
        "mtr_n_desc": mtr_n_desc,
        "mtr_stats_sha1": mtr_stats_sha1,
        "mtr_descriptor_pathway": ("precomputed" if cfg.get("descriptor_precompute_dir") else "live")
                                  if mtr_n_desc else None,
        "host": _host_facts(),
        "total_steps": total_steps,
        "total_forward_passes_target": total_fps,
        "warmup_steps": warmup_steps,
        "augmentation": augmentation,
        "unsupervised_subset_fraction": mlm_subset_fraction,
        "encoder_params": int(sum(p.numel() for p in model.encoder.parameters())),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # ---- training loop ----
    use_amp = train_cfg.bf16 and torch.cuda.is_available()
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if use_amp else {"enabled": False}

    model.train()
    forward_passes_seen = 0
    tokens_seen = 0
    tokens_seen_dev = torch.zeros((), device=device, dtype=torch.long)
    start_time = time.time()
    last_log_time = start_time
    # Truncate ('w', not 'a'): a re-run into an existing run_dir must start metrics.jsonl
    # fresh — a leftover stale file (old mtime) makes the watchdog kill the fresh run for
    # "no recent progress" before the first metric is written.
    metrics_file = metrics_path.open("w", buffering=1)

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

    last_per_family: Dict[str, float] = {}
    try:
        for step, (mode, batch) in enumerate(iterator):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(**autocast_kwargs) if use_amp else _NoCtx():
                if mode == "mlm":
                    loss = model.forward_mlm(batch["input_ids"], batch["attention_mask"], batch["labels"])
                elif mode == "mtr":
                    loss = model.forward_mtr(batch["input_ids"], batch["attention_mask"],
                                             batch["mtr_targets"], train_cfg.mtr_loss)
                else:  # supervised -> (total, per_family)
                    loss, last_per_family = model.forward_sup(
                        batch["input_ids"], batch["attention_mask"], batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            forward_passes_seen += batch["input_ids"].shape[0]
            tokens_seen_dev += batch["attention_mask"].sum()  # on-device; sync only at log

            if (step + 1) % train_cfg.log_every_steps == 0 or step == total_steps - 1:
                now = time.time()
                tokens_seen = int(tokens_seen_dev.item())
                rec = {
                    "step": step + 1,
                    "objective": mode,
                    "loss": float(loss.detach().item()),
                    "lr": scheduler.get_last_lr()[0],
                    "forward_passes_seen": forward_passes_seen,
                    "tokens_seen": tokens_seen,
                    "elapsed_seconds": now - start_time,
                    "step_time_seconds": (now - last_log_time) / train_cfg.log_every_steps,
                    "timestamp": now,
                }
                if last_per_family:
                    rec["sup_per_family"] = {k: round(v, 4) for k, v in last_per_family.items()}
                metrics_file.write(json.dumps(rec) + "\n")
                metrics_file.flush()
                os.fsync(metrics_file.fileno())
                last_log_time = now
                heartbeat("ok")
                extra = f" perfam={rec.get('sup_per_family')}" if "sup_per_family" in rec else ""
                print(f"[step {step+1}/{total_steps}] obj={mode} loss={rec['loss']:.4f} fp={forward_passes_seen}/{total_fps}{extra}", flush=True)

            # Periodic checkpoint for long runs: overwrite the encoder dir so a spot
            # interruption (paired with the launcher's S3-sync sidecar) loses at most
            # save_every_steps of progress. 0 disables (short runs save only at the end).
            if train_cfg.save_every_steps and (step + 1) % train_cfg.save_every_steps == 0:
                model.save_encoder(str(encoder_save_path))
                heartbeat("checkpoint")

        # ---- top-up: the budget is FORWARD PASSES, not steps ----
        # total_steps = total_fps // batch_size assumes every batch is exactly batch_size. It is
        # not: a small `unsupervised_subset_fraction` exhausts its subset many times and each
        # restart yields a short tail batch, so the run finishes all its steps having seen fewer
        # forward passes than its budget. The H1 frac0p001 rung landed at 92% (canonical) and 97%
        # (enumerated) of 2M FP -- an 8% compute shortfall that differed BETWEEN the two arms being
        # compared, i.e. a confound in the very contrast the experiment measures.
        #
        # So: keep drawing batches until the budget is actually met. Runs that already reach it
        # (every full-fraction run, and every wave before this) enter this loop zero times and are
        # bit-identical to before. Top-up steps run at the scheduler's final LR -- the cosine
        # schedule has completed -- which is recorded in metadata rather than hidden.
        topup_steps = 0
        if forward_passes_seen < total_fps:
            cap = max(1, total_steps // 2)          # refuse to run away if batches are tiny
            extra_iter = MultiObjectiveBatchIterator(
                {"mlm": mlm_loader, "mtr": mtr_loader, "supervised": sup_loader},
                objectives, total_batches=cap, seed=pretraining_seed + 10_000,
            )
            print(f"[topup] {forward_passes_seen}/{total_fps} FP after {total_steps} steps "
                  f"({100*forward_passes_seen/total_fps:.1f}%) - continuing at final LR", flush=True)
            for mode, batch in extra_iter:
                if forward_passes_seen >= total_fps:
                    break
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(**autocast_kwargs) if use_amp else _NoCtx():
                    if mode == "mlm":
                        loss = model.forward_mlm(batch["input_ids"], batch["attention_mask"], batch["labels"])
                    elif mode == "mtr":
                        loss = model.forward_mtr(batch["input_ids"], batch["attention_mask"], batch["targets"])
                    else:
                        loss, _ = model.forward_supervised(
                            batch["input_ids"], batch["attention_mask"], batch["targets"], batch["family"])
                    if isinstance(loss, tuple):
                        loss = loss[0]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                forward_passes_seen += batch["input_ids"].shape[0]
                topup_steps += 1
            metrics_file.write(json.dumps({
                "step": total_steps + topup_steps, "topup_steps": topup_steps,
                "forward_passes_seen": forward_passes_seen,
                "total_forward_passes_target": total_fps}) + "\n")
            metrics_file.flush(); os.fsync(metrics_file.fileno())
            print(f"[topup] +{topup_steps} steps -> {forward_passes_seen}/{total_fps} FP "
                  f"({100*forward_passes_seen/total_fps:.1f}%)", flush=True)
        # metadata.json is written before training starts, so persist the top-up count now --
        # otherwise the field is set on a dict nobody ever serialises again.
        metadata["topup_steps"] = topup_steps
        metadata["final_forward_passes_seen"] = forward_passes_seen
        metadata_path.write_text(json.dumps(metadata, indent=2))

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

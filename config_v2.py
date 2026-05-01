"""v2 config dataclasses. One source of truth for training and eval settings."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfigV2:
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 514
    vocab_size: int = 1000
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


@dataclass
class TrainingConfigV2:
    learning_rate: float = 2.0e-4
    batch_size: int = 64
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    mlm_probability: float = 0.15
    total_forward_passes: int = 250_000_000
    seed: int = 42
    bf16: bool = True
    grad_clip: float = 1.0
    log_every_steps: int = 50
    save_every_steps: int = 0
    dataloader_num_workers: int = 2


@dataclass
class EvalConfigV2:
    freeze_encoder: bool = True
    num_epochs: int = 50
    early_stopping_patience: int = 10
    head_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    pretraining_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    cache_encoder_forward: bool = True
    head_learning_rate: float = 1.0e-3
    head_batch_size: int = 64


SUPERVISED_FAMILIES_V2 = ["L1000_MCF7", "L1000_VCAP", "PCBA", "PCQM", "WONG"]
"""Canonical alphabetical-ish ordering. Family-count = N takes the first N."""

MOLECULENET_TASKS_V2 = [
    ("BACE", "classification"),
    ("BBBP", "classification"),
    ("ClinTox", "classification"),
    ("Tox21", "classification"),
    ("ToxCast", "classification"),
    ("SIDER", "classification"),
    ("ESOL", "regression"),
    ("FreeSolv", "regression"),
    ("Lipophilicity", "regression"),
]
"""Pre-registered v2 task subset. Cannot be revised post-hoc."""


@dataclass
class RunSelectionV2:
    """One run's identity + training mode."""
    run_id: str
    run_type: str  # "random_baseline" | "unsup_only" | "sup_only" | "mixed" | "smoke" | "family_order" | "stretch_unsup"
    pretraining_seed: int = 0
    mixing_ratio: float = 1.0  # fraction of batches that are MLM (1.0 = pure MLM, 0.0 = pure sup)
    n_families: int = 0  # 0 = no supervised families; otherwise first N from SUPERVISED_FAMILIES_V2
    family_order: Optional[List[str]] = None  # None = use canonical order; else explicit
    total_forward_passes: Optional[int] = None  # override default budget

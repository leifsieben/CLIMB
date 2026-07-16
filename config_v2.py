"""v2 config dataclasses. One source of truth for training and eval settings.

Backbone is a modern ~40M ModernBERT-style MLM encoder (RoPE, pre-norm, GeGLU,
flash/SDPA) via HuggingFace's drop-in ``ModernBert*`` classes — replacing the
2019-era RoBERTa used in v1. ``build_modernbert_config`` is the single place the
backbone config is constructed so pretraining, evaluation, and the random baseline
all instantiate an identical architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfigV2:
    """ModernBERT backbone hyperparameters. Defaults land at ~41M encoder params
    (verified empirically: hidden 512 / 12 layers / intermediate 1536)."""
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 1536  # ModernBERT GeGLU post-gate dim
    max_position_embeddings: int = 256  # RoPE position cap; >99% of SMILES fit in 256
    vocab_size: int = 1000
    # ModernBERT regularization (kept modest; 0.0 matches the released config).
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    embedding_dropout: float = 0.0
    norm_eps: float = 1e-5
    global_attn_every_n_layers: int = 3
    local_attention: int = 128  # sliding-window size for local-attention layers


def build_modernbert_config(
    model_cfg: "ModelConfigV2",
    vocab_size: int,
    *,
    pad_token_id: int = 1,
    bos_token_id: int = 0,
    eos_token_id: int = 2,
    cls_token_id: Optional[int] = None,
    sep_token_id: Optional[int] = None,
    attn_implementation: str = "sdpa",
):
    """Construct a ``ModernBertConfig`` from a ``ModelConfigV2`` + tokenizer ids.

    Our SMILES tokenizer has no CLS/SEP; position-0 (``<s>``/bos) is used as the
    pooled 'cls' token, so ``cls_token_id`` defaults to bos and ``sep`` to eos.
    These ids are cosmetic for our own pooling but must be < vocab_size.
    """
    from transformers import ModernBertConfig

    cls_id = bos_token_id if cls_token_id is None else cls_token_id
    sep_id = eos_token_id if sep_token_id is None else sep_token_id
    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
        max_position_embeddings=model_cfg.max_position_embeddings,
        attention_dropout=model_cfg.attention_dropout,
        mlp_dropout=model_cfg.mlp_dropout,
        embedding_dropout=model_cfg.embedding_dropout,
        norm_eps=model_cfg.norm_eps,
        global_attn_every_n_layers=model_cfg.global_attn_every_n_layers,
        local_attention=model_cfg.local_attention,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        cls_token_id=cls_id,
        sep_token_id=sep_id,
        # Deterministic + portable across CPU test boxes and GPU workers.
        reference_compile=False,
        attn_implementation=attn_implementation,
    )


@dataclass
class TrainingConfigV2:
    learning_rate: float = 2.0e-4
    batch_size: int = 256  # ModernBERT + SDPA fits a larger batch than v1's 64
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    mlm_probability: float = 0.3  # ModernBERT pretrains at 30% masking
    # Cap training sequence length. >99% of PubChem SMILES fit in 128 tokens; capping
    # here bounds padding memory and keeps throughput high (long-tail SMILES otherwise
    # blow up a batch under masked-mean padding). Separate from the RoPE position cap.
    train_max_length: int = 128
    # Supervised / MTR objectives (Workstream A rebuild):
    supervised_regression_loss: str = "mae"  # MAE for quantum/L1000 (MiniMol); {mae,mse}
    mtr_loss: str = "mse"                     # ChemBERTa-2 MTR on normalized descriptors
    uncertainty_weighting: bool = True        # Kendall homoscedastic weights across heads
    total_forward_passes: int = 250_000_000
    seed: int = 42
    bf16: bool = True
    grad_clip: float = 1.0
    log_every_steps: int = 50
    save_every_steps: int = 0
    dataloader_num_workers: int = 4
    # "canonical" = fast pre-tokenized stream; "enumerated" = raw-SMILES stream with
    # on-the-fly RDKit randomization (see data_v2.StreamingRawSmilesDataset).
    augmentation: str = "canonical"


@dataclass
class EvalConfigV2:
    freeze_encoder: bool = True
    # Frozen-featurizer readout (fixes the v1 CLS-linear-probe pathology).
    pool: str = "mean"           # {cls, mean, cls_mean}
    standardize: str = "zscore"  # {zscore, pca_whiten, none}
    head: str = "mlp"            # {linear, mlp, xgb}
    num_epochs: int = 100
    early_stopping_patience: int = 15
    head_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    pretraining_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    cache_encoder_forward: bool = True
    max_length: int = 256
    # Exp-D label-efficiency: subsample the downstream train split (None = full).
    train_subsample: Optional[int] = None
    subsample_seed: int = 0


SUPERVISED_FAMILIES_V2 = ["PCBA", "L1000_MCF7", "L1000_VCAP", "PCQM"]
"""The four LargeMix families MiniMol used (WONG dropped). PCBA first — it is the
dataset that actually transfers to downstream (MiniMol Table 6)."""

# Named data groups for the dense-vs-sparse ablation (Workstream A experiments).
SUPERVISED_GROUPS = {
    "pcba": ["PCBA"],
    "l1000": ["L1000_MCF7", "L1000_VCAP"],
    "pcqm": ["PCQM"],                                  # negative control (MiniMol)
    "sparse_all": ["PCBA", "L1000_MCF7", "L1000_VCAP"],  # sparse only, no descriptors
    "all": ["PCBA", "L1000_MCF7", "L1000_VCAP", "PCQM"],
    # Faithful MiniMol LargeMix (PCQM + PCBA + L1000) PLUS WONG — the full supervised
    # recipe used as the primary sparse arm in Phase 2 (uncertainty-weighted so PCQM's
    # single-task negative transfer can't dominate the mixture).
    "minimol_full": ["PCBA", "L1000_MCF7", "L1000_VCAP", "PCQM", "WONG"],
}

# Per-family loss weight priors (used directly if uncertainty_weighting is off;
# quantum down-weighted ×1/5 per MiniMol). With uncertainty weighting these are
# just initial scales.
SUPERVISED_FAMILY_WEIGHTS = {
    "PCBA": 1.0, "L1000_MCF7": 1.0, "L1000_VCAP": 1.0, "PCQM": 0.2, "WONG": 1.0,
}

# Per-family row caps for stratified loading so no family starves another. L1000 is
# tiny (~12-15k molecules) so its cap never binds; PCQM is huge + down-weighted.
SUPERVISED_FAMILY_CAPS = {
    "PCBA": 500_000, "L1000_MCF7": 100_000, "L1000_VCAP": 100_000, "PCQM": 200_000, "WONG": 100_000,
}

DESCRIPTOR_STATS_PATH = "configs/descriptor_stats.json"

# Pre-registered v2 downstream suite: one dataset per common cheminformatics use
# case, all ~1-8k. DISJOINT from SUPERVISED_FAMILIES_V2 (no PCBA leakage). Cannot
# be revised post-hoc. Large-scale HIV/QM9 are an optional later extension.
MOLECULENET_TASKS_V2 = [
    ("ESOL", "regression"),          # physical chemistry — aqueous solubility (~1.1k)
    ("BBBP", "classification"),      # ADMET — blood-brain barrier / CNS (~2.0k)
    ("BACE", "classification"),      # drug discovery — beta-secretase binding (~1.5k)
    ("Tox21", "classification"),     # toxicity / safety screening (~7.8k, 12 tasks)
    ("QM7", "regression"),           # quantum mechanics — atomization energy (~7k)
]
"""Pre-registered 5-task suite. Metrics: ROC-AUC (clf), RMSE/MAE (reg)."""


@dataclass
class RunSelectionV2:
    """One run's identity + training mode."""
    run_id: str
    run_type: str  # random_baseline | unsup_only | sup | mtr | mlm_mtr | ablation | scaling | smoke
    pretraining_seed: int = 0
    # Per-batch objective sampler weights, e.g. {"mlm":1.0} / {"mtr":1.0} /
    # {"supervised":1.0} / {"mlm":0.5,"mtr":0.5}. Supersedes mixing_ratio when set.
    objectives: Optional[dict] = None
    mixing_ratio: float = 1.0  # legacy: fraction of MLM batches (1.0 pure MLM, 0.0 pure sup)
    supervised_families: Optional[List[str]] = None  # which families the supervised head uses
    n_families: int = 0
    family_order: Optional[List[str]] = None
    total_forward_passes: Optional[int] = None
    augmentation: str = "canonical"  # {canonical, enumerated}
    unsupervised_subset_fraction: Optional[float] = None  # scaling curve
    init_encoder_path: Optional[str] = None  # warm-start encoder (unsup→sup sequential)

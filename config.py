from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:

    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    max_position_embeddings: int = 256
    vocab_size: int = 1000

    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    embedding_dropout: float = 0.0
    norm_eps: float = 1e-5
    global_attn_every_n_layers: int = 3
    local_attention: int = 128


def build_modernbert_config(
    model_cfg: "ModelConfig",
    vocab_size: int,
    *,
    pad_token_id: int = 1,
    bos_token_id: int = 0,
    eos_token_id: int = 2,
    cls_token_id: Optional[int] = None,
    sep_token_id: Optional[int] = None,
    attn_implementation: str = "sdpa",
):

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

        reference_compile=False,
        attn_implementation=attn_implementation,
    )


@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-4
    batch_size: int = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    mlm_probability: float = 0.3


    train_max_length: int = 128

    supervised_regression_loss: str = "mae"
    mtr_loss: str = "mse"
    uncertainty_weighting: bool = True
    total_forward_passes: int = 250_000_000
    seed: int = 42
    bf16: bool = True
    grad_clip: float = 1.0
    log_every_steps: int = 50
    save_every_steps: int = 0
    dataloader_num_workers: int = 4


    augmentation: str = "canonical"


SUPERVISED_FAMILIES = ["PCBA", "L1000_MCF7", "L1000_VCAP", "PCQM"]


SUPERVISED_FAMILY_WEIGHTS = {
    "PCBA": 1.0, "L1000_MCF7": 1.0, "L1000_VCAP": 1.0, "PCQM": 0.2,
}


SUPERVISED_FAMILY_CAPS = {
    "PCBA": 500_000, "L1000_MCF7": 100_000, "L1000_VCAP": 100_000, "PCQM": 200_000,
}

DESCRIPTOR_STATS_PATH = "configs/descriptor_stats.json"


MOLECULENET_TASKS = [
    ("ESOL", "regression"),
    ("QM7", "regression"),
    ("BACE", "classification"),
    ("Tox21", "classification"),
    ("HIV", "classification"),
]

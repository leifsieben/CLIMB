"""v2 random-encoder baseline.

Instantiates a RobertaModel from the v2 model config with NO pretrained weights
(seeded), saves it to a tmp encoder dir, copies the trained tokenizer alongside
(so we don't accidentally turn this into a tokenizer ablation), then calls eval_v2.

Critical assertion: the saved encoder weights must NOT match any other encoder's
weights (including any cached one from a previous run). We verify by checksumming
the state_dict against a freshly-instantiated reference model with the same seed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import ModernBertModel, set_seed

from config_v2 import ModelConfigV2, MOLECULENET_TASKS_V2, build_modernbert_config
from eval_v2 import evaluate
from storage_utils import materialize_tokenizer_dir


def _state_dict_checksum(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_random_encoder(model_cfg: ModelConfigV2, seed: int, save_dir: str, vocab_size: int = None) -> str:
    cfg = build_modernbert_config(model_cfg, vocab_size or model_cfg.vocab_size)
    set_seed(seed)
    model = ModernBertModel(cfg)
    actual_checksum = _state_dict_checksum(model)

    # Reference: re-create with the same seed; should match.
    set_seed(seed)
    reference = ModernBertModel(cfg)
    ref_checksum = _state_dict_checksum(reference)
    assert actual_checksum == ref_checksum, (
        "Random-init encoder checksum doesn't match a freshly seeded reference; "
        "something contaminated the initialization. Refusing to save."
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    return str(save_path)


def run(
    output_dir: str,
    pretraining_seed: int,
    head_seeds,
    tokenizer_path: str,
    model_cfg: ModelConfigV2,
    pool_mode: str = "mean",
    standardize: str = "zscore",
    head: str = "mlp",
    max_length: int = 256,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve tokenizer + read its vocab size
    tokenizer_local = materialize_tokenizer_dir(tokenizer_path)
    tokenizer_dest = out / "tokenizer"
    tokenizer_dest.mkdir(exist_ok=True)
    for fn in os.listdir(tokenizer_local):
        src = Path(tokenizer_local) / fn
        if src.is_file():
            shutil.copy2(src, tokenizer_dest / fn)
    from transformers import PreTrainedTokenizerFast
    tk = PreTrainedTokenizerFast.from_pretrained(tokenizer_local)
    vocab_size = tk.vocab_size or model_cfg.vocab_size

    # Random encoder
    encoder_dir = out / "encoder"
    make_random_encoder(model_cfg, seed=pretraining_seed, save_dir=str(encoder_dir), vocab_size=vocab_size)

    # Eval — identical pipeline to the trained encoders (this is the floor anchor)
    eval_dir = out / "moleculenet"
    evaluate(
        encoder_path=str(encoder_dir),
        tokenizer_path=str(tokenizer_dest),
        output_dir=str(eval_dir),
        head_seeds=list(head_seeds),
        datasets=MOLECULENET_TASKS_V2,
        featurizer="encoder",
        pool_mode=pool_mode,
        standardize=standardize,
        head=head,
        max_length=max_length,
    )


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    selection = cfg.get("selection", {})
    pretraining_seed = int(selection.get("pretraining_seed", 0))
    eval_cfg = cfg.get("evaluation", {}) or {}
    head_seeds = eval_cfg.get("head_seeds", [0, 1, 2])
    tokenizer_path = cfg["tokenizer_path"]
    model_cfg = ModelConfigV2(**cfg.get("model", {}))

    run(
        args.run_dir, pretraining_seed, head_seeds, tokenizer_path, model_cfg,
        pool_mode=eval_cfg.get("pool", "mean"),
        standardize=eval_cfg.get("standardize", "zscore"),
        head=eval_cfg.get("head", "mlp"),
        max_length=eval_cfg.get("max_length", 256),
    )


if __name__ == "__main__":
    main()

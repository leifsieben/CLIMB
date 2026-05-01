"""Loss-decrease smoke tests for the v2 model.

Verify that on a tiny synthetic batch:
  - MLM forward+backward decreases loss across a few steps.
  - Supervised forward+backward decreases loss across a few steps.
  - The mixed iterator can drive the trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import RobertaConfig

sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain_v2 import ClimbV2Model


def _tiny_config(vocab_size=200) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=0,
    )


def test_mlm_loss_decreases():
    cfg = _tiny_config()
    model = ClimbV2Model(cfg, n_supervised_tasks=0, supervised_task_types=[])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    input_ids = torch.randint(1, 200, (4, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[input_ids < 50] = -100  # only some positions contribute to loss

    losses = []
    for _ in range(20):
        l = model.forward_mlm(input_ids, attention_mask, labels)
        optim.zero_grad()
        l.backward()
        optim.step()
        losses.append(float(l))
    assert losses[-1] < losses[0] * 0.95, f"MLM loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_supervised_loss_decreases():
    cfg = _tiny_config()
    model = ClimbV2Model(cfg, n_supervised_tasks=3, supervised_task_types=["regression"] * 3)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    input_ids = torch.randint(1, 200, (4, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randn(4, 3)

    losses = []
    for _ in range(20):
        l = model.forward_sup(input_ids, attention_mask, labels)
        optim.zero_grad()
        l.backward()
        optim.step()
        losses.append(float(l))
    assert losses[-1] < losses[0] * 0.5, f"Sup loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_supervised_classification_with_nan_mask():
    """Classification + regression mixed columns + NaN labels should still produce a finite loss."""
    cfg = _tiny_config()
    model = ClimbV2Model(cfg, n_supervised_tasks=2, supervised_task_types=["classification", "regression"])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    input_ids = torch.randint(1, 200, (4, 16))
    attention_mask = torch.ones_like(input_ids)
    # Half the rows have only column 0 (classification), half only column 1 (regression).
    labels = torch.full((4, 2), float("nan"))
    labels[0, 0] = 1.0; labels[1, 0] = 0.0
    labels[2, 1] = 0.5; labels[3, 1] = -0.5

    l = model.forward_sup(input_ids, attention_mask, labels)
    assert torch.isfinite(l), f"Loss was non-finite: {l}"
    optim.zero_grad()
    l.backward()
    optim.step()


def test_all_nan_labels_yield_finite_loss():
    cfg = _tiny_config()
    model = ClimbV2Model(cfg, n_supervised_tasks=2, supervised_task_types=["regression", "regression"])
    input_ids = torch.randint(1, 200, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full((2, 2), float("nan"))
    l = model.forward_sup(input_ids, attention_mask, labels)
    assert torch.isfinite(l)

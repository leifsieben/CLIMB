"""Loss-decrease + correctness tests for the v2 multi-objective model.

Covers: MLM, dense MTR, and the per-family supervised multi-head (BCE/MAE, NaN-masked,
uncertainty-weighted), plus the sequential warm-start path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_v2 import ModelConfigV2, build_modernbert_config
from pretrain_v2 import ClimbV2Model


def _tiny_config(vocab_size=200):
    mc = ModelConfigV2(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, max_position_embeddings=64, vocab_size=vocab_size)
    return build_modernbert_config(mc, vocab_size)


def _sup_spec(col_family, task_types, uncertainty=False):
    fams = list(dict.fromkeys(col_family))
    return dict(col_family=col_family, task_types=task_types,
                family_weights={f: 1.0 for f in fams}, regression_loss="mae",
                uncertainty=uncertainty)


def test_mlm_loss_decreases():
    model = ClimbV2Model(_tiny_config())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    ids = torch.randint(1, 200, (4, 16)); mask = torch.ones_like(ids)
    labels = ids.clone(); labels[ids < 50] = -100
    losses = []
    for _ in range(20):
        l = model.forward_mlm(ids, mask, labels)
        optim.zero_grad(); l.backward(); optim.step(); losses.append(float(l))
    assert losses[-1] < losses[0] * 0.95, f"MLM loss stuck: {losses[0]:.3f}->{losses[-1]:.3f}"


def test_mtr_loss_decreases():
    model = ClimbV2Model(_tiny_config(), mtr_n_desc=20)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    ids = torch.randint(1, 200, (4, 16)); mask = torch.ones_like(ids)
    targets = torch.randn(4, 20)
    losses = []
    for _ in range(30):
        l = model.forward_mtr(ids, mask, targets, "mse")
        optim.zero_grad(); l.backward(); optim.step(); losses.append(float(l))
    assert losses[-1] < losses[0] * 0.6, f"MTR loss stuck: {losses[0]:.3f}->{losses[-1]:.3f}"


def test_supervised_multihead_decreases():
    spec = _sup_spec(["A", "A", "B"], ["regression", "regression", "regression"])
    model = ClimbV2Model(_tiny_config(), supervised=spec)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    ids = torch.randint(1, 200, (4, 16)); mask = torch.ones_like(ids)
    labels = torch.randn(4, 3)
    losses = []
    for _ in range(30):
        l, per_family = model.forward_sup(ids, mask, labels)
        optim.zero_grad(); l.backward(); optim.step(); losses.append(float(l))
    assert set(per_family) == {"A", "B"}
    assert losses[-1] < losses[0] * 0.7, f"sup loss stuck: {losses[0]:.3f}->{losses[-1]:.3f}"


def test_supervised_mixed_types_nan_masked():
    """Classification + regression families with NaN labels → finite loss + backward."""
    spec = _sup_spec(["PCBA", "L1000"], ["classification", "regression"], uncertainty=True)
    model = ClimbV2Model(_tiny_config(), supervised=spec)
    ids = torch.randint(1, 200, (4, 8)); mask = torch.ones_like(ids)
    labels = torch.tensor([[1.0, 0.5], [0.0, float("nan")], [float("nan"), -0.5], [1.0, 0.2]])
    l, per_family = model.forward_sup(ids, mask, labels)
    assert torch.isfinite(l) and set(per_family) == {"PCBA", "L1000"}
    l.backward()


def test_warm_start_loads_encoder(tmp_path):
    m1 = ClimbV2Model(_tiny_config())
    enc = tmp_path / "encoder"
    m1.save_encoder(str(enc))
    m2 = ClimbV2Model(_tiny_config())
    m2.load_init_encoder(str(enc))
    for (k, a), (_, b) in zip(m1.encoder.state_dict().items(), m2.encoder.state_dict().items()):
        assert torch.equal(a, b), f"warm-start weight mismatch at {k}"

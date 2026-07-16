"""Tests for the v2 frozen-featurizer readout: pooling, standardization, heads,
SMILES enumeration, and the ~40M ModernBERT param count."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import featurize_v2 as F
import heads_v2 as H
from config_v2 import ModelConfigV2, build_modernbert_config
from smiles_augment import randomize_smiles


def test_mean_pool_excludes_padding():
    h = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
    pooled = F.pool(h, mask, "mean")
    assert torch.allclose(pooled[0], h[0, :3].mean(0), atol=1e-5)
    assert torch.allclose(pooled[1], h[1, 0], atol=1e-5)  # single real token
    assert F.pool(h, mask, "cls_mean").shape == (2, 16)


def test_zscore_standardizer_fit_on_train():
    rng = np.random.default_rng(0)
    tr = rng.normal(3.0, 2.0, size=(64, 10))
    params = F.fit_standardizer(tr, "zscore")
    z = F.apply_standardizer(tr, params)
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1.0) < 1e-2


def test_ecfp4_shapes_and_bad_smiles():
    fp = F.ecfp4_features(["CCO", "c1ccccc1", "not_a_smiles["], n_bits=128)
    assert fp.shape == (3, 128)
    assert fp[2].sum() == 0.0  # unparseable → zero row


def test_smiles_enumeration_same_molecule():
    from rdkit import Chem
    s = "CC(=O)Oc1ccccc1C(=O)O"
    variants = {randomize_smiles(s) for _ in range(20)}
    assert len(variants) > 1  # actually enumerates
    canon = {Chem.MolToSmiles(Chem.MolFromSmiles(v)) for v in variants}
    assert len(canon) == 1  # all the same molecule


def test_heads_nan_masked_multitask():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 12)).astype(np.float32)
    Y = (X[:, :3] > 0).astype(np.float32)
    Y[Y[:, 0] == 1, 1] = np.nan  # inject missing labels in a column
    for head in ("linear", "mlp", "xgb"):
        hd = H.make_head(head, "classification", 3, 0).fit(X[:80], Y[:80], X[80:100], Y[80:100])
        auc = H.compute_metric(hd.predict(X[100:]), Y[100:], "classification")
        assert np.isfinite(auc)


def test_modernbert_param_count_near_40m():
    from transformers import ModernBertModel
    cfg = build_modernbert_config(ModelConfigV2(), vocab_size=1000)
    n = sum(p.numel() for p in ModernBertModel(cfg).parameters())
    assert 35e6 < n < 46e6, f"encoder params {n/1e6:.1f}M outside ~40M target"

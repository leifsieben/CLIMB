"""Regression test for the Tox21 missing-label bug.

DeepChem encodes MISSING multitask labels as y=0 with w=0 -- NOT as NaN. The
eval_v2 loader used to return only `.y`, so the 16,012 missing Tox21 entries were
indistinguishable from true inactives and were fed as real negatives into head
training, validation early-stopping, ROC-AUC, NEF and the paired tests. The whole
downstream pipeline masks missing labels by NaN, so `_load_moleculenet` now sets
`y[w==0]=NaN`; these tests lock the exact counts so the bug cannot silently return.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from eval_v2 import _load_moleculenet


def _all_y(name):
    tr_s, tr_y, va_s, va_y, te_s, te_y = _load_moleculenet(name)
    return np.concatenate([tr_y, va_y, te_y], axis=0)


def test_tox21_missing_labels_are_nan():
    """Tox21: 7,823 molecules x 12 endpoints = 93,876 cells; 16,012 missing."""
    y = _all_y("Tox21")
    assert y.shape == (7823, 12)
    n_missing = int(np.isnan(y).sum())
    n_valid = int((~np.isnan(y)).sum())
    n_pos = int((y == 1).sum())   # NaN == 1 is False, so this counts real positives only
    n_neg = int((y == 0).sum())   # NaN == 0 is False, so this counts real negatives only

    assert n_valid == 77864, n_valid
    assert n_pos == 5858, n_pos
    assert n_neg == 72006, n_neg
    assert n_missing == 16012, n_missing
    assert n_pos + n_neg == n_valid
    assert n_valid + n_missing == y.size == 93876


@pytest.mark.parametrize("name", ["BACE", "BBBP", "ESOL", "QM7"])
def test_single_task_datasets_have_no_missing(name):
    """Single-task datasets have all-ones w, so the w==0->NaN mask must be a no-op."""
    y = _all_y(name)
    assert not np.isnan(y).any(), f"{name} unexpectedly has missing (NaN) labels after masking"

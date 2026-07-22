"""Tests for the end-to-end fine-tuned eval arm.

The properties that matter here are not "does it train" (a GPU wave answers that) but
"can its output be trusted to pair with the frozen arms". Three things silently destroy
that and none of them raise:

  - a label mask that lets NaN entries into the loss (Tox21 is ~83% NaN),
  - a task whose type is defaulted rather than declared (QM7 trained as classification),
  - a suite_summary key that does not match what the figures read (HIV's headline is
    `HIV_nef1_MEAN`, not `HIV_MEAN`).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_v2 import MOLECULENET_TASKS_V2
from finetune_e2e_v2 import _TokenizedSet, _masked_loss, _write_outputs


class _FakeTokenizer:
    """Character-level stand-in: keeps the test off the real tokenizer files."""
    pad_token_id = 7

    def __call__(self, smiles, truncation=True, max_length=8):
        return {"input_ids": [[ord(c) % 50 + 10 for c in s][:max_length] for s in smiles]}


def test_tokenized_set_pads_to_longest_in_batch_and_masks_padding():
    tok = _TokenizedSet(_FakeTokenizer(), ["CC", "CCCCC", "C"], max_length=8)
    ids, mask = tok.batch([0, 1, 2], torch.device("cpu"))
    assert ids.shape == (3, 5) and mask.shape == (3, 5)
    assert mask.sum(dim=1).tolist() == [2, 5, 1]
    # padding must be the tokenizer's pad id, not 0 — a wrong pad id is a real token
    assert (ids[2, 1:] == _FakeTokenizer.pad_token_id).all()


def test_tokenized_set_truncates_at_max_length():
    tok = _TokenizedSet(_FakeTokenizer(), ["C" * 40], max_length=8)
    ids, mask = tok.batch([0], torch.device("cpu"))
    assert ids.shape[1] == 8 and int(mask.sum()) == 8


def test_masked_loss_ignores_nan_labels():
    logits = torch.zeros(4, 2)
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0],
                      [float("nan"), float("nan")], [1.0, 0.0]])
    loss = _masked_loss(logits, y, is_clf=True)
    assert loss is not None and torch.isfinite(loss)
    # 5 valid entries, all logit 0 -> BCE = ln 2 exactly. Any NaN leaking in makes this NaN.
    assert abs(float(loss) - float(np.log(2))) < 1e-5


def test_masked_loss_returns_none_when_every_label_is_nan():
    # An all-NaN Tox21 batch must contribute NO gradient rather than a NaN one.
    assert _masked_loss(torch.zeros(2, 3), torch.full((2, 3), float("nan")), is_clf=True) is None


def test_every_registered_task_has_a_declared_type():
    # finetune_v2 used to resolve types with .get(ds, "classification"), so a regression
    # task that was never listed trained with BCE and was reported as ROC-AUC.
    import finetune_v2
    for name, task_type in MOLECULENET_TASKS_V2:
        assert finetune_v2.TASK_TYPE[name] == task_type
    assert finetune_v2.TASK_TYPE["QM7"] == "regression"


def test_suite_summary_keys_match_the_frozen_arm_schema():
    rows = [
        {"dataset": "HIV", "task_type": "classification", "main_metric": "roc_auc",
         "head_seed": "MEAN", "main_value": 0.7, "n_train": 1, "featurizer": "encoder_finetune",
         "pool": "mean", "standardize": "none", "head": "linear_e2e", "elapsed_seconds": 0.0},
        {"dataset": "HIV", "task_type": "classification", "main_metric": "nef1",
         "head_seed": "MEAN", "main_value": 0.4, "n_train": 1, "featurizer": "encoder_finetune",
         "pool": "mean", "standardize": "none", "head": "linear_e2e", "elapsed_seconds": 0.0},
        {"dataset": "ESOL", "task_type": "regression", "main_metric": "rmse",
         "head_seed": "STD", "main_value": 0.05, "n_train": 1, "featurizer": "encoder_finetune",
         "pool": "mean", "standardize": "none", "head": "linear_e2e", "elapsed_seconds": 0.0},
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_outputs(out, rows)
        suite = json.loads((out / "suite_summary.json").read_text())
    # The primary metric keeps the bare key; the secondary is namespaced. Fig A1 reads
    # TASKS["HIV"]["suite_key"] == "HIV_nef1", so both must be present and distinct.
    assert suite["HIV_MEAN"] == 0.7
    assert suite["HIV_nef1_MEAN"] == 0.4
    assert suite["ESOL_STD"] == 0.05


def test_summary_csv_columns_are_identical_to_the_frozen_arm():
    import csv as _csv
    rows = [{"dataset": "ESOL", "task_type": "regression", "featurizer": "encoder_finetune",
             "pool": "mean", "standardize": "none", "head": "linear_e2e", "main_metric": "rmse",
             "head_seed": "MEAN", "n_train": 10, "main_value": 0.5, "elapsed_seconds": 1.0}]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _write_outputs(out, rows)
        with (out / "moleculenet_summary.csv").open() as f:
            header = next(_csv.reader(f))
    assert header == ["dataset", "task_type", "featurizer", "pool", "standardize", "head",
                      "main_metric", "head_seed", "n_train", "main_value", "elapsed_seconds"]

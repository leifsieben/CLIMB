"""Tests for v2 data loaders.

Critical regression test: each row in the supervised parquet must be yielded
exactly once per epoch (the v1 sharding bug was duplicating rows by num_workers).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_v2 import (
    MLMCollator,
    SupervisedCollator,
    SupervisedInRAMDataset,
    load_supervised_inram,
    MixedModeBatchIterator,
)


def _make_tiny_supervised_parquet(tmp_path: Path, n_rows: int = 50, n_label_cols: int = 4) -> Path:
    """Tiny synthetic parquet with input_ids, attention_mask, and labels."""
    rng = np.random.default_rng(0)
    rows = {
        "input_ids": [rng.integers(0, 1000, size=10).tolist() for _ in range(n_rows)],
        "attention_mask": [[1] * 10 for _ in range(n_rows)],
    }
    fams = ["FAMA", "FAMB"]
    for fam in fams:
        for i in range(n_label_cols // 2):
            col = f"{fam}__task_{i}"
            rows[col] = [float(rng.uniform()) if rng.random() > 0.3 else None for _ in range(n_rows)]

    tbl = pa.table(rows)
    out = tmp_path / "tiny.parquet"
    pq.write_table(tbl, out)
    return out


def test_supervised_each_row_once():
    """Loading the supervised parquet should yield exactly one row per molecule
    (per epoch) regardless of DataLoader num_workers. This is the v1 regression."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        parquet = _make_tiny_supervised_parquet(tdp, n_rows=50)
        ids, mask, labels, cols, types = load_supervised_inram(
            str(parquet), families=["FAMA", "FAMB"], max_length=10,
        )
        n_rows = ids.shape[0]
        assert n_rows > 0 and n_rows <= 50  # some may be dropped if all-NaN
        ds = SupervisedInRAMDataset(ids, mask, labels, types)

        for nw in [0, 2, 4]:
            loader = DataLoader(
                ds, batch_size=4, shuffle=False, num_workers=nw,
                collate_fn=SupervisedCollator(pad_token_id=0), drop_last=False,
            )
            seen = []
            for batch in loader:
                seen.extend(batch["input_ids"][:, 0].tolist())  # use position 0 as a stand-in row id
            # Map-style dataset yields each idx exactly once; PyTorch DataLoader handles sharding.
            assert len(seen) == n_rows, f"num_workers={nw}: got {len(seen)} rows, expected {n_rows}"


def test_supervised_drops_all_nan_rows():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        rows = {
            "input_ids": [[1, 2, 3]] * 5,
            "attention_mask": [[1, 1, 1]] * 5,
            "FAMA__task_0": [None, 0.5, None, 1.0, None],
        }
        out = tdp / "p.parquet"
        pq.write_table(pa.table(rows), out)
        ids, mask, labels, cols, types = load_supervised_inram(
            str(out), families=["FAMA"], max_length=3,
        )
        assert ids.shape[0] == 2  # only rows with labels survive


def test_supervised_task_type_inference():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        rows = {
            "input_ids": [[1, 2, 3]] * 6,
            "attention_mask": [[1, 1, 1]] * 6,
            "FAMA__binary_task": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "FAMA__regression_task": [0.1, 0.7, 0.5, 0.3, 0.9, 0.2],
        }
        out = tdp / "p.parquet"
        pq.write_table(pa.table(rows), out)
        _, _, _, cols, types = load_supervised_inram(
            str(out), families=["FAMA"], max_length=3,
        )
        type_map = dict(zip(cols, types))
        assert type_map["FAMA__binary_task"] == "classification"
        assert type_map["FAMA__regression_task"] == "regression"


def test_mlm_collator_basic():
    coll = MLMCollator(mask_token_id=4, vocab_size=1000, mlm_probability=0.15, pad_token_id=0)
    examples = [
        {"input_ids": [10, 20, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1]},
        {"input_ids": [11, 22, 33], "attention_mask": [1, 1, 1]},
    ]
    out = coll(examples)
    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["labels"].shape == (2, 5)
    # Padding positions must have label=-100
    assert (out["labels"][1, 3:] == -100).all()


def test_mixed_iterator_ratio():
    """MixedModeBatchIterator yields modes at the configured ratio (within tolerance)."""
    fake_mlm = [{"input_ids": torch.zeros(4, 5, dtype=torch.long),
                 "attention_mask": torch.ones(4, 5, dtype=torch.long),
                 "labels": torch.zeros(4, 5, dtype=torch.long)} for _ in range(100)]
    fake_sup = [{"input_ids": torch.zeros(4, 5, dtype=torch.long),
                 "attention_mask": torch.ones(4, 5, dtype=torch.long),
                 "labels": torch.zeros(4, 3)} for _ in range(100)]

    class FakeLoader:
        def __init__(self, batches): self.batches = batches
        def __iter__(self): return iter(self.batches)

    it = MixedModeBatchIterator(
        mlm_loader=FakeLoader(fake_mlm),
        sup_loader=FakeLoader(fake_sup),
        mixing_ratio=0.5,
        total_batches=2000,
        seed=0,
    )
    counts = {"mlm": 0, "sup": 0}
    for mode, _ in it:
        counts[mode] += 1
    ratio = counts["mlm"] / 2000
    assert abs(ratio - 0.5) < 0.05, f"mixed ratio {ratio} too far from 0.5"


def test_mixed_iterator_pure_mlm_no_sup_loader():
    fake_mlm = [{"input_ids": torch.zeros(4, 5, dtype=torch.long),
                 "attention_mask": torch.ones(4, 5, dtype=torch.long),
                 "labels": torch.zeros(4, 5, dtype=torch.long)} for _ in range(50)]
    class FakeLoader:
        def __init__(self, batches): self.batches = batches
        def __iter__(self): return iter(self.batches)

    it = MixedModeBatchIterator(
        mlm_loader=FakeLoader(fake_mlm), sup_loader=None,
        mixing_ratio=1.0, total_batches=100, seed=0,
    )
    modes = [mode for mode, _ in it]
    assert all(m == "mlm" for m in modes)
    assert len(modes) == 100


# ---------- E13 / H2c corrupted-pretraining control ----------

def test_corrupted_collator_shuffle_tokens_preserves_pairing_and_pins_special():
    """shuffle_tokens must destroy ORDER only: CLS/SEP stay pinned, the interior is a
    permutation of the original tokens, and each masked slot still asks for its own
    original token (input_ids and labels permuted together). If this breaks, the E13
    control silently stops being a valid control."""
    from data_v2 import CorruptedCollator
    CLS, SEP, PAD, MASK = 1, 2, 0, 3

    class Base:
        def __call__(self, _ex):
            return {
                "input_ids": torch.tensor([[CLS, 10, 11, 12, 13, 14, SEP, PAD],
                                           [CLS, 20, MASK, 22, SEP, PAD, PAD, PAD]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0],
                                                [1, 1, 1, 1, 1, 0, 0, 0]]),
                "labels": torch.tensor([[-100] * 8,
                                        [-100, -100, 21, -100, -100, -100, -100, -100]]),
            }

    out = CorruptedCollator(Base(), "shuffle_tokens", seed=0)(None)
    ids, labels = out["input_ids"], out["labels"]
    # CLS / SEP pinned, interior is a permutation of the originals
    assert ids[0, 0].item() == CLS and ids[0, 6].item() == SEP
    assert sorted(ids[0, 1:6].tolist()) == [10, 11, 12, 13, 14]
    assert sorted(ids[1, 1:4].tolist()) == sorted([20, MASK, 22])
    # (input, label) correspondence survives the permutation
    assert any(ids[1, p].item() == MASK and labels[1, p].item() == 21 for p in range(1, 4))


def test_corrupted_collator_shuffle_targets_permutes_rows():
    """shuffle_targets must break the molecule->descriptor mapping while keeping the
    target distribution identical (rows permuted, never the identity)."""
    from data_v2 import CorruptedCollator

    class Base:
        def __call__(self, _ex):
            return {"mtr_targets": torch.arange(12, dtype=torch.float32).reshape(4, 3)}

    orig = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    got = CorruptedCollator(Base(), "shuffle_targets", seed=0)(None)["mtr_targets"]
    assert sorted(map(tuple, got.tolist())) == sorted(map(tuple, orig.tolist()))
    assert not torch.equal(got, orig)


def test_corrupted_collator_rejects_unknown_mode():
    from data_v2 import CorruptedCollator
    with pytest.raises(ValueError):
        CorruptedCollator(object(), "not_a_mode")

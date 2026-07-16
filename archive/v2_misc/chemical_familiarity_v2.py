"""v2 chemical-familiarity analysis (Figure 5).

For each MoleculeNet test molecule, compute whether its canonical SMILES is in:
  - the unsupervised pretraining corpus, AND/OR
  - the supervised pretraining corpus.

Produces a 2x2 confusion matrix per (dataset × pretraining_condition):
  Group A: in unsup AND in sup
  Group B: in unsup ONLY
  Group C: in sup ONLY
  Group D: in NEITHER

Per group, compute mean test metric across the runs in the condition. Output a
heatmap + CSV.

Tanimoto-decile binning is intentionally NOT in v0 of this script — it lives behind
a `--mode tanimoto` flag and is gated on the 2x2 result showing a meaningful split.

Usage:
    python chemical_familiarity_v2.py --manifest <manifest.json> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


# ---------- canonical SMILES ----------

def _canonicalize_smiles(smiles_iter) -> Set[str]:
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("rdkit required: pip install rdkit-pypi")
    out: Set[str] = set()
    for s in smiles_iter:
        if not s:
            continue
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            out.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
        except Exception:
            continue
    return out


# ---------- corpus loaders ----------

def load_unsup_corpus_smiles(parquet_path: str, max_rows: Optional[int] = None) -> Set[str]:
    """Read SMILES from the filtered unsupervised parquet.
    For tokenized corpora that don't keep SMILES, fall back to the raw filtered parquet.
    """
    import pyarrow.dataset as pads
    from storage_utils import is_s3_uri, parquet_dataset

    ds = parquet_dataset(parquet_path) if is_s3_uri(parquet_path) else pads.dataset(parquet_path, format="parquet")
    candidates = ["smiles", "SMILES", "canonical_smiles"]
    schema_cols = ds.schema.names
    smiles_col = next((c for c in candidates if c in schema_cols), None)
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {parquet_path}; columns: {schema_cols[:10]}...")

    smiles_acc: List[str] = []
    rows = 0
    for batch in ds.to_batches(columns=[smiles_col], batch_size=8192):
        smiles_acc.extend(batch.column(0).to_pylist())
        rows += len(batch)
        if max_rows and rows >= max_rows:
            break

    return _canonicalize_smiles(smiles_acc)


def load_supervised_corpus_smiles(parquet_path: str) -> Set[str]:
    """The supervised wide parquet keeps SMILES; just canonicalize the column."""
    import pyarrow.dataset as pads
    from storage_utils import is_s3_uri, parquet_dataset

    ds = parquet_dataset(parquet_path) if is_s3_uri(parquet_path) else pads.dataset(parquet_path, format="parquet")
    candidates = ["smiles", "SMILES", "canonical_smiles"]
    schema_cols = ds.schema.names
    smiles_col = next((c for c in candidates if c in schema_cols), None)
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {parquet_path}")

    smiles_acc: List[str] = []
    for batch in ds.to_batches(columns=[smiles_col], batch_size=8192):
        smiles_acc.extend(batch.column(0).to_pylist())
    return _canonicalize_smiles(smiles_acc)


# ---------- per-molecule eval ----------

def molecule_groups(test_smiles: List[str], unsup: Set[str], sup: Set[str]) -> List[str]:
    """For each test SMILES, return one of A/B/C/D."""
    canon = []
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("rdkit required: pip install rdkit-pypi")

    out = []
    for s in test_smiles:
        if not s:
            out.append("D"); continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append("D"); continue
        c = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        in_u = c in unsup
        in_s = c in sup
        out.append("A" if in_u and in_s else "B" if in_u else "C" if in_s else "D")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unsup_parquet", required=True,
                   help="Filtered unsup parquet path (with SMILES column)")
    p.add_argument("--sup_parquet", required=True,
                   help="Supervised wide parquet (with SMILES column)")
    p.add_argument("--output", default="experiments/robust_matrix_v2/familiarity")
    p.add_argument("--max_unsup_rows", type=int, default=None)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("[familiarity] loading unsup corpus...")
    unsup_set = load_unsup_corpus_smiles(args.unsup_parquet, max_rows=args.max_unsup_rows)
    print(f"  |unsup_set| = {len(unsup_set)}")

    print("[familiarity] loading sup corpus...")
    sup_set = load_supervised_corpus_smiles(args.sup_parquet)
    print(f"  |sup_set| = {len(sup_set)}")

    # Save sets as plain text (canonical-SMILES, one per line) for downstream use.
    (out / "unsup_canonical_smiles.txt").write_text("\n".join(sorted(unsup_set)))
    (out / "sup_canonical_smiles.txt").write_text("\n".join(sorted(sup_set)))

    # Per MoleculeNet task: load test SMILES, group, save group counts.
    from config_v2 import MOLECULENET_TASKS_V2
    from eval_v2 import _load_moleculenet  # reuse loader

    summary = {}
    for ds_name, _ in MOLECULENET_TASKS_V2:
        try:
            _, _, _, _, te_smiles, te_y = _load_moleculenet(ds_name)
        except Exception as exc:
            print(f"  skip {ds_name}: {exc}")
            continue
        groups = molecule_groups(te_smiles, unsup_set, sup_set)
        counts = {g: groups.count(g) for g in "ABCD"}
        summary[ds_name] = {
            "groups": groups,
            "counts": counts,
            "test_y": te_y.tolist() if hasattr(te_y, "tolist") else list(te_y),
            "test_smiles": list(te_smiles),
        }
        print(f"  {ds_name}: {counts}")

    (out / "molecule_groups.json").write_text(json.dumps(summary, indent=2))
    print(f"[familiarity] wrote {out / 'molecule_groups.json'}")


if __name__ == "__main__":
    main()

"""SI Fig e — build the canonical-vs-augmented SMILES table (the figure's ONLY input).

Does it help to pretrain on ENUMERATED (randomised) SMILES rather than the single RDKit-canonical
string per molecule? Augmentation is standard practice in SMILES language models, so a null result
is worth stating. Wave `climb_v2` scaling ladder:

  canonical   one RDKit-canonical SMILES per molecule
  enumerated  randomised SMILES writings of the same molecules (same molecule set, more strings)

Five corpus fractions (0.001, 0.01, 0.1, 0.3, full) x 3 pretraining seeds each, so the comparison
is made at matched corpus fraction all the way up the ladder — augmentation could plausibly help
most where data is scarce, and this design can see that.

PANEL SCOPE — MoleculeACE and hERG only. Those are the two canonical panels where all 30 runs were
scored with 3 pretraining seeds. BACE/Tox21/QM7 exist for these arms ONLY as a single-seed, single
hold-out eval under climb_v2/<arm>/moleculenet/ — a different protocol from the 5-fold CV every
other figure uses, and one seed rather than three. Mixing that in would put two protocols in one
figure, so those panels are emitted empty rather than filled with a number that cannot be compared.
CBS was never run for this wave.

  MoleculeACE  chemeleon_suite/moleculeace/<run>/results.csv -> macro RMSE over 30 targets, per
               (pretraining seed x eval seed); value = mean, sd = SD across the 3 pretraining-seed
               macro-means
  hERG         chemeleon_suite/polaris/<run>/polaris_scores.csv (tdcommons/herg, roc_auc) -> mean
               over 3 pretraining seeds x 3 eval seeds; sd = SD across the 3 pretraining-seed means

Writes: figure_data/SI_Fig_e/SI_Fig_e_augmentation.csv
Run:    python3 scripts/build_SI_Fig_e_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "SI_Fig_e" / "SI_Fig_e_augmentation.csv"

MODES = [("canonical", "canonical"), ("enumerated", "augmented")]
FRACTIONS = [("0p001", 0.001), ("0p01", 0.01), ("0p1", 0.1), ("0p3", 0.3), ("full", 1.0)]
SEEDS = ["s0", "s1", "s2"]
HERG = ("tdcommons/herg", "roc_auc")


def mace_macro(run):
    """Macro RMSE over the 30 targets for one run, averaged over its eval seeds."""
    p = FD / "chemeleon_suite" / "moleculeace" / run / "results.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    o = d[(d.subset == "overall") & (d.metric == "rmse")]
    return float(o.groupby("seed").value.mean().mean()) if len(o) else np.nan


def herg_mean(run):
    p = FD / "chemeleon_suite" / "polaris" / run / "polaris_scores.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    v = d[(d.task == HERG[0]) & (d.metric == HERG[1])].value.astype(float)
    return float(v.mean()) if len(v) else np.nan


def main() -> None:
    rows = []
    for panel, fn, higher in (("MoleculeACE", mace_macro, 0), ("hERG", herg_mean, 1)):
        for mode_key, mode_label in MODES:
            for frac_key, frac in FRACTIONS:
                per_seed = [fn(f"scaling_{mode_key}_frac{frac_key}_{s}") for s in SEEDS]
                per_seed = [v for v in per_seed if np.isfinite(v)]
                if not per_seed:
                    continue
                rows.append(dict(panel=panel, higher_better=higher, mode=mode_label,
                                 fraction=frac, frac_key=frac_key,
                                 value=round(float(np.mean(per_seed)), 6),
                                 sd=(round(float(np.std(per_seed, ddof=1)), 6)
                                     if len(per_seed) > 1 else ""),
                                 n_seeds=len(per_seed)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "higher_better", "mode", "fraction", "frac_key", "value", "sd", "n_seeds"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")
    print("panels drawn empty (protocol mismatch / not run): CBS, BACE, Tox21, QM7")

    d = pd.DataFrame(rows)
    print("\ndoes augmentation beat canonical at matched corpus fraction?")
    print("  (delta signed so + = augmented better; compared against the pooled seed SD)")
    for panel in ("MoleculeACE", "hERG"):
        p = d[d.panel == panel]
        sign = 1 if p.higher_better.iloc[0] else -1
        print(f"\n   {panel}:")
        for _, frac in FRACTIONS:
            c = p[(p["mode"] == "canonical") & (p.fraction == frac)]
            a = p[(p["mode"] == "augmented") & (p.fraction == frac)]
            if not len(c) or not len(a):
                continue
            delta = sign * (float(a.value.iloc[0]) - float(c.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(c.sd, errors="coerce").iloc[0],
                          pd.to_numeric(a.sd, errors="coerce").iloc[0])
            flag = "*" if np.isfinite(sd) and abs(delta) > sd else " "
            print(f"      frac {frac:<6} canonical {float(c.value.iloc[0]):9.4f}   "
                  f"augmented {float(a.value.iloc[0]):9.4f}   delta {delta:+8.4f}{flag}")
    print("\n   * = |delta| exceeds the combined seed SD")


if __name__ == "__main__":
    main()

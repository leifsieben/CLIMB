"""SI Fig d — build the canonical-vs-augmented SMILES table (the figure's ONLY input).

Does it help to pretrain on ENUMERATED (randomised) SMILES rather than the single RDKit-canonical
string per molecule? Augmentation is standard practice in SMILES language models, so a null result
is worth stating. Wave `climb_v2` scaling ladder:

  canonical   one RDKit-canonical SMILES per molecule
  enumerated  randomised SMILES writings of the same molecules (same molecule set, more strings)

Five corpus fractions (0.001, 0.01, 0.1, 0.3, full) x 3 pretraining seeds each, so the comparison
is made at matched corpus fraction all the way up the ladder — augmentation could plausibly help
most where data is scarce, and this design can see that.

PANEL SCOPE — ALL SIX canonical panels, 3 pretraining seeds each.

CORRECTION 2026-08-17: an earlier version of this script filled only MoleculeACE and hERG and drew
the other four empty, on the finding that BACE/Tox21/QM7 existed only as a single-seed single
hold-out. That was a WRONG-ROOT error. `figure_data/climb_v2/` is the ROUND-1 wave, which never
saved its encoders and only ever produced the single hold-out; the retrained wave that every other
figure uses is `figure_data/climb_v2_h1/`, and there all 30 runs have the 5-fold scaffold CV. CBS
was likewise present all along, under figure_data/cbs_benchmark/<run>/moleculenet_cv/ — it was
absent only from experiment_cbs/cbs_nef1_summary.csv, a deprecated precomputed file whose ARMS list
never included this wave (the aggregator itself stopped reading it for the same reason).

Resolve MolNet paths through climb_v2_h1 and CBS through cbs_benchmark; never through climb_v2 or
the cbs summary CSV.

  MoleculeACE  chemeleon_suite/moleculeace/<run>/results.csv -> macro RMSE over 30 targets, per
               (pretraining seed x eval seed); value = mean, sd = SD across the 3 pretraining-seed
               macro-means
  hERG         chemeleon_suite/polaris/<run>/polaris_scores.csv (tdcommons/ames, roc_auc) -> mean
               over 3 pretraining seeds x 3 eval seeds; sd = SD across the 3 pretraining-seed means

Writes: figure_data/SI_fig_d/SI_fig_d_augmentation.csv
Run:    python3 scripts/build_SI_fig_d_table.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for figures.sixpanel

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "SI_fig_d" / "SI_fig_d_augmentation.csv"

MODES = [("canonical", "canonical"), ("enumerated", "augmented")]
H1 = FD / "climb_v2_h1"                     # the RETRAINED wave (climb_v2 is round-1: do not use)
CBS_ROOT = FD / "cbs_benchmark"
# HIV added 2026-08-19. It took CBS's canonical slot in the panel swap and this list was not
# updated with it, so SI fig d drew an EMPTY HIV panel and printed "not run on this protocol" --
# while all 30 climb_v2_h1/scaling_* runs had HIV on disk the whole time. Same shape as the
# hardcoded lists in build_SI_fig_b_table and six_panel_scaling.
MOL_PANELS = {"BACE": "BACE_MEAN", "Tox21": "Tox21_MEAN", "QM7": "QM7_MEAN",
              "HIV": "HIV_nef1_MEAN"}
FRACTIONS = [("0p001", 0.001), ("0p01", 0.01), ("0p1", 0.1), ("0p3", 0.3), ("full", 1.0)]
SEEDS = ["s0", "s1", "s2"]
AMES = ("tdcommons/ames", "roc_auc")


def _suite(root, run, key):
    """Read `key` from a run's suite summary, preferring the per-task corrected subdir.

    `key` is "<task>_MEAN" for the MolNet panels, so the task is recoverable and the same subdir
    preference the figures use (figures.sixpanel.NATIVE_SUBDIRS: qm7native, tox21fixed) applies
    here too. Anything else falls back to moleculenet_cv/.
    """
    from figures.sixpanel import NATIVE_SUBDIRS
    task = key[:-5] if key.endswith("_MEAN") else None
    for sub in NATIVE_SUBDIRS.get(task, ("moleculenet_cv",)):
        p = root / run / sub / "suite_summary.json"
        if p.exists():
            v = json.load(open(p)).get(key)
            if v is not None:
                return float(v)
    return np.nan


def molnet(run, key):
    """One MolNet panel's 5-fold CV mean for a run, from the RETRAINED wave."""
    return _suite(H1, run, key)


def cbs(run):
    """CBS NEF1% for a run. `cbs_MEAN` is ROC-AUC; the panel metric is `cbs_nef1_MEAN`."""
    return _suite(CBS_ROOT, run, "cbs_nef1_MEAN")


def mace_macro(run):
    """Macro RMSE over the 30 targets for one run, averaged over its eval seeds."""
    p = FD / "chemeleon_suite" / "moleculeace" / run / "results.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    o = d[(d.subset == "overall") & (d.metric == "rmse")]
    return float(o.groupby("seed").value.mean().mean()) if len(o) else np.nan


def ames_mean(run):
    p = FD / "chemeleon_suite" / "polaris" / run / "polaris_scores.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    v = d[(d.task == AMES[0]) & (d.metric == AMES[1])].value.astype(float)
    return float(v.mean()) if len(v) else np.nan


def main() -> None:
    rows = []
    panels = [("MoleculeACE", mace_macro, 0), ("Ames", ames_mean, 1), ("CBS", cbs, 1)]
    panels += [(p, (lambda k: (lambda run: molnet(run, k)))(k), 0 if p == "QM7" else 1)
               for p, k in MOL_PANELS.items()]
    for panel, fn, higher in panels:
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
    d0 = pd.DataFrame(rows)
    filled = sorted(set(d0.panel))
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")
    print(f"panels filled: {', '.join(filled)}")

    d = pd.DataFrame(rows)
    print("\ndoes augmentation beat canonical at matched corpus fraction?")
    print("  (delta signed so + = augmented better; compared against the pooled seed SD)")
    for panel in filled:
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

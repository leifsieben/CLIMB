from __future__ import annotations

import csv, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FD = ROOT / "figure_data"
OUT = FD / "SI_fig_b" / "SI_fig_b_vocab.csv"


RUNS = [("bpe_261", "BPE", 261), ("bpe_1000", "BPE", 1000),
        ("bpe_3000", "BPE", 3000), ("bpe_12000", "BPE", 12000),
        ("unigram_261", "Unigram", 261), ("unigram_700", "Unigram", 700),
        ("unigram_1200", "Unigram", 1200), ("unigram_3000", "Unigram", 3000)]


MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "HIV": "nef1"}
AMES = ("tdcommons/ames", "roc_auc")
CBS_ROOT = FD / "cbs_benchmark"

from figures.arms import PANEL_ORDER as _PO
PANELS = list(_PO) + ["CBS"]
HIGHER = {"MoleculeACE": 0, "CBS": 1, "BACE": 1, "Ames": 1, "Tox21": 1, "QM7": 0, "HIV": 1}


def mace(run):
    """(macro RMSE over 30 targets, SD across the 3 eval-seed macro-means, n)."""
    p = FD / "chemeleon_suite" / "moleculeace" / run / "results.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    o = d[(d.subset == "overall") & (d.metric == "rmse")]
    per_seed = o.groupby("seed").value.mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)), len(per_seed)


def cbs(run):

    import json as _json
    p = CBS_ROOT / run / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return None
    j = _json.load(open(p))
    m = j.get("cbs_nef1_MEAN")
    if m is None:
        return None
    sd = j.get("cbs_nef1_STD")

    n = 5
    sd = float(sd) * np.sqrt(n / (n - 1)) if sd is not None else np.nan
    return float(m), sd, n


def ames(run):
    p = FD / "chemeleon_suite" / "polaris" / run / "polaris_scores.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    v = d[(d.task == AMES[0]) & (d.metric == AMES[1])].value.astype(float)
    if not len(v):
        return None
    return float(v.mean()), float(v.std(ddof=1)), len(v)


def main() -> None:
    cv = pd.read_csv(FD / "climb_v2_vocab" / "vocab_cv_summary.csv")
    rows = []
    for run, family, vocab in RUNS:
        def add(panel, res, n_label):
            if res is None:
                return
            m, sd, n = res
            rows.append(dict(panel=panel, higher_better=HIGHER[panel], family=family,
                             vocab=vocab, run=run, value=round(m, 6),
                             sd=("" if not np.isfinite(sd) else round(sd, 6)),
                             n=n, n_kind=n_label))

        add("MoleculeACE", mace(run), "eval seeds")
        add("Ames", ames(run), "eval seeds")
        add("CBS", cbs(run), "CV folds")
        for panel, metric in MOL_PANELS.items():
            g = cv[(cv.run == run) & (cv.dataset == panel) & (cv.metric == metric)]
            if len(g):
                add(panel, (float(g["mean"].iloc[0]), float(g.fold_std.iloc[0]), 5), "CV folds")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["panel", "higher_better", "family", "vocab", "run", "value", "sd", "n", "n_kind"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    d = pd.DataFrame(rows)
    empty = [p for p in PANELS if p not in set(d.panel)]
    print(f"wrote {OUT.relative_to(ROOT)}  {len(rows)} rows")
    print(f"panels with no vocab-wave run (drawn empty): {', '.join(empty) or 'none'}")


    print("\nspread across vocab sizes vs the median replicate SD (the near-null test):")
    print(f"   {'panel':<12}{'family':<9}{'min':>10}{'max':>10}{'range':>10}{'median sd':>11}  verdict")
    for panel in PANELS:
        for fam in ("BPE", "Unigram"):
            g = d[(d.panel == panel) & (d.family == fam)]
            if len(g) < 2:
                continue
            rng = g.value.max() - g.value.min()
            sd = pd.to_numeric(g.sd, errors="coerce").median()
            verdict = "within noise" if rng <= sd else f"{rng/sd:.1f}x noise"
            print(f"   {panel:<12}{fam:<9}{g.value.min():>10.4f}{g.value.max():>10.4f}"
                  f"{rng:>10.4f}{sd:>11.4f}  {verdict}")


if __name__ == "__main__":
    main()

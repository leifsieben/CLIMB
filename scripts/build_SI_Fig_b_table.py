"""SI Fig b — build the tokenizer/vocabulary-size table (the figure's ONLY input).

Wave `climb_v2_vocab` (README §7.2): two tokenizer families — byte-level BPE (the main-paper
family) and Unigram-LM — each at four reachable, distinct vocabulary sizes. SMILES tokenization
saturates, so those four are the whole reachable range. One MLM-only encoder per tokenizer, all at
2M forward passes, same corpus, same frozen-probe eval as everything else.

CONFOUND, disclosed not removed: the embedding auto-sizes to the vocabulary, so parameter count
grows with vocab (~41.0M -> 47.1M). Vocabulary size and embedding parameters cannot be separated
in this design.

COMPUTE NOTE: these encoders are 2M FP, NOT the 8M of the mainline arms, so their absolute values
are NOT comparable to Fig A2/B. The figure deliberately carries no mainline reference line for
that reason — the comparison that matters here is BPE vs Unigram and across vocab size, all at
matched compute.

Panels: the canonical six. CBS has no vocab-wave run for any arm and is emitted empty.

  MoleculeACE  chemeleon_suite/moleculeace/<run>/results.csv  -> macro RMSE over 30 targets,
               mean over the 3 eval seeds; sd = SD across the 3 eval-seed macro-means
  hERG         chemeleon_suite/polaris/<run>/polaris_scores.csv (tdcommons/herg, roc_auc)
               -> mean over the 3 eval seeds; sd = SD across them
  BACE/Tox21/QM7  climb_v2_vocab/vocab_cv_summary.csv -> 5-fold CV mean; sd = fold_std
  CBS          not run

ERROR BARS ARE THE POINT of this figure. The finding is a near-null — vocabulary size moves the
frozen-probe score by less than the replicate noise almost everywhere — which cannot be stated
without showing that noise. This is why SI Fig b carries error bars while Figs B and F do not.

Writes: figure_data/SI_Fig_b/SI_Fig_b_vocab.csv
Run:    python3 scripts/build_SI_Fig_b_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
OUT = FD / "SI_Fig_b" / "SI_Fig_b_vocab.csv"

# (run dir, family, actual vocabulary size)
RUNS = [("bpe_261", "BPE", 261), ("bpe_1000", "BPE", 1000),
        ("bpe_3000", "BPE", 3000), ("bpe_12000", "BPE", 12000),
        ("unigram_261", "Unigram", 261), ("unigram_700", "Unigram", 700),
        ("unigram_1200", "Unigram", 1200), ("unigram_3000", "Unigram", 3000)]

MOL_PANELS = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
HERG = ("tdcommons/herg", "roc_auc")
PANELS = ["MoleculeACE", "CBS", "BACE", "hERG", "Tox21", "QM7"]
HIGHER = {"MoleculeACE": 0, "CBS": 1, "BACE": 1, "hERG": 1, "Tox21": 1, "QM7": 0}


def mace(run):
    """(macro RMSE over 30 targets, SD across the 3 eval-seed macro-means, n)."""
    p = FD / "chemeleon_suite" / "moleculeace" / run / "results.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    o = d[(d.subset == "overall") & (d.metric == "rmse")]
    per_seed = o.groupby("seed").value.mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)), len(per_seed)


def herg(run):
    p = FD / "chemeleon_suite" / "polaris" / run / "polaris_scores.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    v = d[(d.task == HERG[0]) & (d.metric == HERG[1])].value.astype(float)
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
        add("hERG", herg(run), "eval seeds")
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

    # is any vocab effect larger than the replicate noise it sits in?
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

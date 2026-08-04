"""Standalone SI figures for the vocabulary-size scaling law (wave climb_v2_vocab).

Reads the eight frozen-probe 5-fold CV summaries under figure_data/climb_v2_vocab/ and produces:

  figures_vocab/figSV_vocab_scaling.{png,pdf}   -- six task panels, downstream metric vs actual
                                                    tokenizer vocab (log x), BPE vs Unigram, fold-std bars.
  figures_vocab/figSV_vocab_effect.{png,pdf}    -- one summary panel: per-task change from the vocab-261
                                                    baseline in units of that task's fold std ("effect
                                                    size in noise units"), which makes the near-null result
                                                    legible at a glance.

Deliberately writes to figures_vocab/ (NOT figures_out/) so it does not disturb the notebook session's
figures_out<->notebook sync. Style mirrors the paper's rcParams for visual consistency.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("figure_data/climb_v2_vocab")
OUT = Path("figures_vocab"); OUT.mkdir(exist_ok=True)

# actual (measured) vocab per run -> the x-axis; nominal target is irrelevant once trained
RUNS = {
    "bpe":     [(261, "bpe_261"), (1000, "bpe_1000"), (3000, "bpe_3000"), (12000, "bpe_12000")],
    "unigram": [(261, "unigram_261"), (700, "unigram_700"), (876, "unigram_1200"), (3000, "unigram_3000")],
}
FAMILY = {"bpe": dict(color="#4477AA", marker="o", label="Byte-level BPE"),
          "unigram": dict(color="#EE7733", marker="s", label="Unigram-LM")}
# (dataset, metric, pretty title, lower_is_better)
PANELS = [
    ("ESOL",  "rmse",    "ESOL — RMSE ↓ (solubility)",        True),
    ("QM7",   "rmse",    "QM7 — RMSE ↓ (atomization E)",      True),
    ("BBBP",  "roc_auc", "BBBP — ROC-AUC ↑",                  False),
    ("BACE",  "roc_auc", "BACE — ROC-AUC ↑",                  False),
    ("Tox21", "roc_auc", "Tox21 — ROC-AUC ↑",                 False),
    ("HIV",   "nef1",    "HIV — NEF1% ↑ (virtual screen)",    False),
]

def read(run, ds, metric, stat):
    for r in csv.DictReader(open(DATA / f"{run}_cv.csv")):
        if r["dataset"] == ds and r["main_metric"] == metric and r["head_seed"] == stat:
            return float(r["main_value"])
    return None

def series(fam, ds, metric):
    xs, ys, es = [], [], []
    for v, run in RUNS[fam]:
        m, s = read(run, ds, metric, "MEAN"), read(run, ds, metric, "STD")
        if m is not None:
            xs.append(v); ys.append(m); es.append(s if s is not None else 0.0)
    return np.array(xs, float), np.array(ys, float), np.array(es, float)

def _style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.grid": True, "grid.color": "#B0B0B0", "grid.linewidth": 0.5, "grid.alpha": 0.35,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 120,
    })

def fig_panels():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, (ds, metric, title, lower) in zip(axes.ravel(), PANELS):
        for fam, sty in FAMILY.items():
            x, y, e = series(fam, ds, metric)
            ax.errorbar(x, y, yerr=e, color=sty["color"], marker=sty["marker"], ms=5.5,
                        lw=1.6, capsize=2.5, elinewidth=0.9, label=sty["label"], zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([261, 1000, 3000, 12000]); ax.set_xticklabels(["261", "1k", "3k", "12k"])
        ax.set_title(title)
        ax.set_xlabel("tokenizer vocab (actual, log)")
        ax.margins(x=0.08)
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Vocabulary-size scaling for unsupervised (MLM) pretraining — frozen-probe 5-fold scaffold CV\n"
                 "(2M forward passes, same corpus & eval; error bars = ± fold std)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figSV_vocab_scaling.{ext}")
    plt.close(fig)

def fig_effect():
    """Largest-vocab change from the vocab-261 baseline, in units of that task's fold std."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    dodge = {"bpe": -0.16, "unigram": +0.16}
    for fam, sty in FAMILY.items():
        for k, (ds, metric, title, lower) in enumerate(PANELS):
            x, y, e = series(fam, ds, metric)
            # improvement of the LARGEST vocab vs the 261 baseline; + = better (down for RMSE, up else)
            delta = (y[0] - y[-1]) if lower else (y[-1] - y[0])
            denom = e[-1] if e[-1] > 0 else np.nan
            eff = delta / denom
            xp = k + dodge[fam]
            ax.plot([xp, xp], [0, eff], color=sty["color"], lw=2.2, zorder=2)
            ax.plot([xp], [eff], color=sty["color"], marker=sty["marker"], ms=7,
                    label=(sty["label"] if k == 0 else None), zorder=3)
    ax.axhspan(-1, 1, color="#888888", alpha=0.13, zorder=0)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels([p[0] for p in PANELS])
    ax.set_ylabel("improvement of largest vocab vs vocab-261\n(in fold-std units)")
    ax.set_title("Does enlarging the vocabulary help? Largest reachable vocab vs the character-level (261) baseline\n"
                 "Shaded band = ±1 fold std — markers inside it are within evaluation noise", fontsize=9.2)
    ax.set_ylim(-2.6, 2.6)
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figSV_vocab_effect.{ext}")
    plt.close(fig)

if __name__ == "__main__":
    _style()
    fig_panels()
    fig_effect()
    print(f"wrote figSV_vocab_scaling + figSV_vocab_effect (png+pdf) -> {OUT}/")

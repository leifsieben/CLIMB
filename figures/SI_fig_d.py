"""SI Fig d — canonical vs augmented (enumerated) SMILES in pretraining.

ONE script, ONE figure: figures_v2/SI_fig_e.png / .pdf

Randomised ("enumerated") SMILES are standard practice in SMILES language models: the same molecule
written many ways is meant to teach the model that the string is arbitrary and the graph is not.
This asks whether it actually buys anything. Wave `climb_v2`: canonical (one RDKit-canonical string
per molecule) vs enumerated, at five corpus fractions x 3 pretraining seeds, so the two are matched
at every rung of the ladder — augmentation could plausibly matter most where data is scarce, and
this design would see that.

THE RESULT IS NOT A NULL, AND IT SPLITS BY TASK. Augmentation helps on four panels and hurts on
one:

  CBS          the largest effect anywhere: +0.09 to +0.14 NEF1% at full corpus and at the sparsest
               fractions, beyond the seed SD at three of five rungs. On a rare-active screen,
               seeing each molecule written many ways evidently matters.
  MoleculeACE  helps consistently, at every corpus fraction (macro RMSE 0.784 -> 0.768 at full
               corpus; +0.015 to +0.019 across the ladder, every point beyond the seed SD). Flat in
               corpus size — a constant offset, not a small-data crutch that washes out.
  BACE, QM7    within noise at almost every rung; no usable effect either way.
  hERG         augmentation HURTS (0.753 -> 0.697 at full corpus, up to -0.099). CAVEAT: hERG has
               132 test molecules and its sampling uncertainty is far larger than its seed SD
               suggests (see the A2 caption) — read the direction, not the magnitude.

So augmentation is worth doing for potency regression and rare-active screening, is neutral on the
plain classification/regression panels, and is the one thing that looks actively harmful on hERG.
That is a more useful answer than the "free win" the practice is usually adopted as.

Error bars are +-1 SD across the 3 PRETRAINING seeds. They are drawn because the claim is about
whether a difference clears the noise; the build script prints that test explicitly.

PANEL SCOPE: all six canonical panels, 3 pretraining seeds each. An earlier version filled only
MoleculeACE and hERG — that was a wrong-root error (climb_v2 is the round-1 wave; the retrained
wave every other figure uses is climb_v2_h1, and CBS lives under cbs_benchmark/ rather than the
deprecated cbs summary CSV). See scripts/build_SI_fig_d_table.py.

Data: figure_data/SI_fig_d/SI_fig_d_augmentation.csv, built by scripts/build_SI_fig_d_table.py.

Run:  python3 scripts/build_SI_fig_d_table.py && python3 -m figures.SI_fig_d
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font
from figures.arms import PANELS, PANEL_ORDER, SHADES
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_d" / "SI_fig_d_augmentation.csv")

# the two corpus variants, split by lightness + marker
# NOT the unsup blues: SI fig b is already a two-line plot over these same six panels in that
# family, and side by side the two figures were indistinguishable. These are the scheme's crimson
# shades, used here purely as the standard red-vs-blue contrast against SI fig b — this figure
# contains no model arms, so red carries no "supervised" meaning in it. (A moss/olive pair was
# tried first and rejected as ugly, 2026-08-17.)
MODES = [("canonical", SHADES["sup"][0], "o"), ("augmented", SHADES["sup"][2], "D")]
YMARGIN = 0.22


def main():
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 5.1))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        ax.set_xlabel("pretraining corpus fraction", fontsize=FS["annot"], color=INK)
        ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if g_all.empty:
            ax.text(0.5, 0.5, "not run on this\nprotocol", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lo, hi = np.inf, -np.inf
        for mode, colour, marker in MODES:
            g = g_all[g_all["mode"] == mode].sort_values("fraction")
            if g.empty:
                continue
            sd = pd.to_numeric(g.sd, errors="coerce").fillna(0).to_numpy()
            ax.errorbar(g.fraction, g.value, yerr=sd, color=colour, ls="-", lw=STYLE["lw"],
                        marker=marker, ms=4.6, mec="white", mew=0.6,
                        elinewidth=1.0, capsize=2.2, capthick=1.1, ecolor=INK, zorder=3)
            lo = min(lo, (g.value - sd).min())
            hi = max(hi, (g.value + sd).max())

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.001, 0.01, 0.1, 1.0]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: {0.001: "0.1%", 0.01: "1%", 0.1: "10%", 1.0: "100%"}.get(v, f"{v:g}")))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(0.0006, 1.7)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=c, marker=m, ms=4.5, lw=1.2, label=lab)
               for lab, c, m in MODES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "SI_fig_d")
    plt.close(fig)

    print("\nSI Fig d — augmented minus canonical (+ = augmented better):")
    for p in PANEL_ORDER:
        g_all = DF[DF.panel == p]
        if g_all.empty:
            print(f"   {p:<12} — not run on this protocol")
            continue
        sign = 1 if g_all.higher_better.iloc[0] else -1
        cells = []
        for frac in sorted(g_all.fraction.unique()):
            c = g_all[(g_all["mode"] == "canonical") & (g_all.fraction == frac)]
            a = g_all[(g_all["mode"] == "augmented") & (g_all.fraction == frac)]
            if not len(c) or not len(a):
                continue
            delta = sign * (float(a.value.iloc[0]) - float(c.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(c.sd, errors="coerce").iloc[0],
                          pd.to_numeric(a.sd, errors="coerce").iloc[0])
            cells.append(f"{frac:>6g}:{delta:+8.4f}{'*' if np.isfinite(sd) and abs(delta) > sd else ' '}")
        print(f"   {p:<12} " + " ".join(cells))
    print("   * = |delta| exceeds the combined pretraining-seed SD")


if __name__ == "__main__":
    main()

"""SI Fig b — does tokenizer family or vocabulary size matter?

ONE script, ONE figure: figures_v2/figSIb.png / .pdf

Wave `climb_v2_vocab` (README §7.2). Two tokenizer families — byte-level BPE (the main-paper
family) and Unigram-LM — each at four reachable, distinct vocabulary sizes. SMILES tokenization
saturates, so those four span the whole reachable range. One MLM-only encoder per tokenizer, all
at matched 2M forward passes, same corpus, same frozen-probe eval as every other arm. Each panel
is one benchmark of the canonical six; x is the ACTUAL vocabulary reached (log scale).

THE RESULT IS A NEAR-NULL, and that is the finding — state it plainly rather than hunting for a
trend. Across the reachable range, vocabulary size moves the frozen-probe score by less than the
replicate noise on 8 of the 10 family x panel combinations, and BPE ~= Unigram at matched vocab.
The character-level floor (vocab 261, no merges) is already competitive everywhere. The two
exceptions are both BPE and both worth one hedged sentence, not a headline: MoleculeACE (range
0.0089 vs sd 0.0018) and hERG (range 0.065 vs sd 0.020) — and hERG's own whisker understates its
true uncertainty badly (132 test molecules; see the A2 caption), so that one should not be leaned
on at all.

ERROR BARS ARE THE POINT here, which is why this figure carries them while Figs B and F do not: a
claim of "within noise" is unreadable without the noise. They are +-1 SD of the panel's replicate
unit — 5 CV folds for BACE/Tox21/QM7, 3 eval seeds for MoleculeACE and hERG.

CONFOUND, disclosed not removed: the embedding auto-sizes to the vocabulary, so parameter count
grows with it (~41.0M -> 47.1M). Vocabulary size and embedding parameters cannot be separated in
this design.

COMPUTE NOTE: 2M FP, NOT the 8M of the mainline arms — absolute values are not comparable to
Fig A2/B, and no mainline reference line is drawn for that reason. The comparison that matters is
internal: family vs family, vocab vs vocab, at matched compute.

PANEL SCOPE: CBS is drawn EMPTY — no vocab-wave arm was ever run on it.

Data: figure_data/figSI/figSIb_vocab.csv, built by scripts/build_SI_fig_b_table.py.

Run:  python3 scripts/build_SI_fig_b_table.py && python3 -m figures.SI_fig_b
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

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_b" / "SI_fig_b_vocab.csv")

# the two tokenizer families; both are unsupervised (MLM) encoders, so both take the unsup family
# hue, separated by lightness and marker rather than by an unrelated colour.
FAMILIES = [("BPE", SHADES["unsup"][0], "o"), ("Unigram", SHADES["unsup"][2], "D")]
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
        ax.set_xlabel("tokenizer vocabulary", fontsize=FS["annot"], color=INK)
        ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if g_all.empty:                       # panel kept in place; the gap is the message
            ax.text(0.5, 0.5, "not run in the\nvocabulary wave", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lo, hi = np.inf, -np.inf
        for fam, colour, marker in FAMILIES:
            g = g_all[g_all.family == fam].sort_values("vocab")
            if g.empty:
                continue
            sd = pd.to_numeric(g.sd, errors="coerce").fillna(0).to_numpy()
            ax.errorbar(g.vocab, g.value, yerr=sd, color=colour, ls="-", lw=STYLE["lw"],
                        marker=marker, ms=4.6, mec="white", mew=0.6,
                        elinewidth=1.0, capsize=2.2, capthick=1.1, ecolor=INK, zorder=3)
            lo = min(lo, (g.value - sd).min())
            hi = max(hi, (g.value + sd).max())

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator([261, 1000, 3000, 12000]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(200, 16000)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=c, marker=m, ms=4.5, lw=1.2, label=f)
               for f, c, m in FAMILIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "SI_fig_b")
    plt.close(fig)

    print("\nSI Fig b — vocabulary spread vs replicate noise (the near-null test):")
    print(f"   {'panel':<12}{'family':<9}{'range over vocab':>18}{'median sd':>12}  verdict")
    for p in PANEL_ORDER:
        g_all = DF[DF.panel == p]
        if g_all.empty:
            print(f"   {p:<12}{'—':<9}{'not run in the vocabulary wave':>32}")
            continue
        for fam, _, _ in FAMILIES:
            g = g_all[g_all.family == fam]
            if len(g) < 2:
                continue
            rng = g.value.max() - g.value.min()
            sd = pd.to_numeric(g.sd, errors="coerce").median()
            print(f"   {p:<12}{fam:<9}{rng:>18.4f}{sd:>12.4f}  "
                  f"{'within noise' if rng <= sd else f'{rng/sd:.1f}x noise'}")


if __name__ == "__main__":
    main()

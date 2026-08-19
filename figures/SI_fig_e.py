"""SI Fig e — where does end-to-end training overtake a pretrained frozen encoder?

ONE script, ONE figure: figures_v2/figF.png / .pdf

What it shows
-------------
Each panel is one benchmark of the canonical six; x is the number of LABELLED training molecules
(log scale), y is absolute downstream performance. Three models, identical hold-out split,
identical label fractions, identical seed grid — the only thing that differs is the model:

  no pretrain, end2end  — the whole network trained from a random init on the downstream task only
  supervised, dense     — pretrained encoder, FROZEN, probe trained on the labels
  unsupervised          — pretrained encoder, FROZEN, probe trained on the labels

The case for pretraining is the small-data end: with a few hundred labels the frozen encoders
should be far ahead, and end2end should only catch up once labels are plentiful. Reading the
crossings:

  MoleculeACE and Ames  the arms SEPARATE as labels increase rather than converging: on
         MoleculeACE all three sit at ~1.17 macro RMSE with 1.9k labels and end at 0.775 / 0.777
         (pretrained) vs 0.905 (end2end) with 38.9k. Whatever pretraining supplies here is not a
         head start that fine-tuning erases -- it is a ceiling that end2end does not reach.
  BACE   pretraining wins at EVERY size — end2end never catches up inside the range (0.725 vs
         0.825 at full data). This is the panel where pretraining pays.
  HIV    the textbook shape, and the largest small-data gap in the figure: at 1.6k labels both
         frozen encoders sit at 0.444 NEF1% against end2end's 0.339 (+31% relative). End2end has
         closed most of it by 3.3k and passes `unsupervised` by 33k, but `supervised, dense` stays
         ahead throughout (0.675 vs 0.611 at full data). Pretraining buys a real head start on a
         rare-active screen and, for the supervised objective, keeps it.
  Tox21  end2end closes the gap and passes `supervised, dense` at full data (0.730 vs 0.722),
         though `unsupervised` still leads (0.736). The advantage of pretraining is spent by ~6k
         labels.
  QM7    end2end is AHEAD at every size below full data; the frozen unsupervised probe is poor on
         this task throughout (212.7 RMSE at full data). Pretraining does not pay here at all.

So the honest answer is task-dependent: pretraining buys a large, durable margin on BACE and HIV, a
margin that expires around a few thousand labels on Tox21, and nothing on QM7. Note that the two
panels where it pays are the two where the label budget is smallest relative to the difficulty of
the task — BACE tops out at 1.2k labels, and HIV is 3.5% active.

NO error bars (matching Fig B, user decision 2026-08-17). The per-point SD across the seed cells is
kept in figure_data/figF/figF_crossover.csv if a referee asks.

PANEL SCOPE: MoleculeACE, CBS and hERG are drawn EMPTY — the label-fraction sweep was only ever run
on MoleculeNet, so no arm has a fraction curve there. The panels are kept in place rather than
silently reshaping the figure to the three tasks that have data; the evals are requested.

PROTOCOL NOTE: single hold-out split, NOT the 5-fold scaffold CV of Figs A2/B, so absolute values
are not comparable across those figures. Internally consistent, which is what the crossing needs.

Data: figure_data/figF/figF_crossover.csv, built by scripts/build_SI_fig_e_table.py.

Run:  python3 scripts/build_SI_fig_e_table.py && python3 -m figures.SI_fig_e
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS, PANELS, PANEL_ORDER
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_e" / "SI_fig_e_crossover.csv")

# same three-line set in every panel; colour comes from arms.py (single source of truth)
LINES = ["e2e_no_pretrain", "sup_dense", "unsup"]
MARKER = {"e2e_no_pretrain": "o", "sup_dense": "s", "unsup": "D"}

YMARGIN = 0.18


def _fmt_n(v, _):
    return f"{v/1000:.1f}k" if v >= 1000 else f"{v:g}"


def main():
    # 2x3 at FULL page width. One row of six was tried and reverted (user 2026-08-19: "too
    # extreme... they become super distorted") -- six panels across 6.69in leaves ~1.05in
    # each, taller than they are wide, which squashes the curves. 2x3 gives ~2.0in panels.
    # The height saving comes from tighter spacing and ONE shared x-axis label instead of
    # six, not from collapsing the grid. Width is ~3.5% over col2 because savefig("tight")
    # trims back to about the text block.
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.7))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"
        # a substituted panel is titled by the dataset actually drawn, never by the panel slot
        title = g_all.task.iloc[0] if not g_all.empty else d["label"]
        ax.set_title(f"{title} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        # x label drawn ONCE under the row, below -- six identical copies of the same
        # words cost a line of height each and say nothing new.
        pass
        ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if g_all.empty:                       # panel kept in place; the gap is the message
            ax.text(0.5, 0.5, "no label-fraction\nsweep run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sub = g_all.substituted_for.iloc[0] if "substituted_for" in g_all else ""
        if isinstance(sub, str) and sub:
            ax.text(0.98, 0.04, f"stand-in for {sub}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=FS["annot"] - 1, color=STYLE["mute"], style="italic")

        lo, hi = np.inf, -np.inf
        for arm in LINES:
            g = g_all[g_all.arm == arm].sort_values("n_train")
            if g.empty:
                continue
            ax.plot(g.n_train, g.value, color=ARMS[arm]["color"], ls="-", lw=STYLE["lw"],
                    marker=MARKER[arm], ms=4.6, mec="white", mew=0.6, zorder=3)
            lo = min(lo, g.value.min())
            hi = max(hi, g.value.max())

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_n))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        n = sorted(g_all.n_train.unique())
        ax.xaxis.set_major_locator(ticker.FixedLocator(n))
        ax.set_xlim(n[0] * 0.78, n[-1] * 1.28)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=ARMS[a]["color"], marker=MARKER[a], ms=4.5, lw=1.2,
                      label=ARMS[a]["label"]) for a in LINES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052),
               ncol=3, fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.155, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.108, "labelled training molecules", ha="center", va="bottom",
             fontsize=FS["annot"], color=INK)
    save(fig, "SI_fig_e")
    plt.close(fig)

    print("\nSI Fig e — absolute performance vs labelled training size:")
    for p in PANEL_ORDER:
        g_all = DF[DF.panel == p]
        if g_all.empty:
            print(f"   {p:<12} — no label-fraction sweep run")
            continue
        if g_all.task.iloc[0] != p:
            print(f"   [{p} panel draws {g_all.task.iloc[0]} — CBS cannot be subsampled]")
        n = sorted(g_all.n_train.unique())
        print(f"   {p} ({g_all.metric.iloc[0]}):   " + "".join(f"{x:>10}" for x in n))
        for arm in LINES:
            g = g_all[g_all.arm == arm].sort_values("n_train")
            print(f"      {ARMS[arm]['label']:<22}" + "".join(f"{v:>10.4f}" for v in g.value))


if __name__ == "__main__":
    main()

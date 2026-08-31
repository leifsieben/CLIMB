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

THE XGBoost ANCHOR (added 2026-08-29, Leif: "just to see that XGBoost actually beats the models at
any dataset size"). It does, on FOUR OF THE SIX PANELS, at every budget:
  MoleculeACE  best at all 5 budgets, and by a wide margin (1.05 -> 0.67 macro RMSE against the
               best model line's 1.16 -> 0.78). Never overtaken.
  QM7          best at all 5. Never overtaken.
  Tox21        best at every budget it shares with the models. Never overtaken.
  HIV          best at every shared budget. Never overtaken.
  Ames         best at 4 of 5; supervised edges it at 582 labels and it retakes the lead after.
  BACE         the ONE panel it never leads -- both CLMs are ahead from 60 labels upward.
So on five of six panels the fingerprint is ahead at nearly every label budget we measured, and on
four it is never beaten at all. BACE is the single exception, and BACE is also the smallest panel
here (1.2k labels at full data). The usual case for pretraining is that it pays when labels are
scarce; on this plate the scarce-label end is where the fingerprint is strongest.

X POSITIONS DIFFER BY <0.1% ON Tox21 AND HIV, and the cause is worth recording rather than hiding:
the three model arms were swept in an environment whose RDKit parsed 7,830 Tox21 and 41,126 HIV
molecules, the anchor in the pinned one that parses 7,823 and 41,120 -- the same canonicalization
drift that cost the fig_B eval boxes a re-run. It puts the anchor at n_train 6,258 where the models
sit at 6,264 (Tox21) and 32,896 against 32,901 (HIV); BACE and QM7 align exactly. Six molecules in
6,264 is 0.10%, invisible on a log axis and far below the spread between arms, and the model arms
are NOT re-run for it (Leif: a measured number stands unless the EVALUATION was wrong).

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

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX
from figures.arms import ARMS, PANELS, PANEL_ORDER
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_e" / "SI_fig_e_crossover.csv")

# same three-line set in every panel; colour comes from arms.py (single source of truth)
# The XGBoost anchor joins as a FOURTH line (Leif 2026-08-29), because the question this figure is
# usually asked about -- does pretraining pay at small label budgets -- has a second half: does any
# of it beat a fingerprint at ANY label budget. Its sweep covers the four MolNet tasks, so its line
# is short on MoleculeACE and Ames rather than absent; the caption must say so.
LINES = ["e2e_no_pretrain", "sup_dense", "unsup", "ecfp_desc"]
MARKER = {"e2e_no_pretrain": "o", "sup_dense": "s", "unsup": "D", "ecfp_desc": "^"}
_unmarked = [a for a in LINES if a not in MARKER]
assert not _unmarked, f"LINES has {_unmarked} with no marker -- add one to MARKER"

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
            # ERROR BARS PER POINT (Leif 2026-08-29): "it looks too much like CLIMB is overtaking
            # XGBoost where in practice it just is within noise". Without them the lines cross and
            # a reader reads a crossing as a result. They do overlap: on BACE at 605 labels
            # unsupervised is 0.814 +/- 0.023 against ECFP4+desc's 0.789 +/- 0.023.
            #
            # +/- 1 SD over the seed cells -- 3 subsample seeds x 3 head seeds = 9, dropping to 3
            # at 100% where there is nothing to subsample, so the LAST point of every line rests on
            # fewer cells than the rest. That is a property of the design, not of the arm.
            #
            # NaN passes through: matplotlib omits it, so a missing spread draws no whisker rather
            # than a zero-length one, which would claim perfect precision.
            e = g.sd.to_numpy(dtype=float)
            ax.errorbar(g.n_train, g.value, yerr=e, fmt="none",
                        ecolor=ARMS[arm]["color"], elinewidth=0.7, capsize=1.6, capthick=0.7,
                        zorder=2)
            # limits follow the WHISKERS, not the points, or the caps clip at the panel edge
            lo = min(lo, np.nanmin(np.where(np.isnan(e), g.value, g.value - e)))
            hi = max(hi, np.nanmax(np.where(np.isnan(e), g.value, g.value + e)))

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
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.066),
               ncol=row_ncol(handles), fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    # Axes -> shared x-label -> legend, each about one text-height apart (user 2026-08-19:
    # "too much white space"). The legend is anchored just BELOW the x-label rather than
    # near the canvas floor: with loc="upper center" a low anchor hangs the legend body off
    # the canvas, and savefig("tight") then GROWS the image downward to contain it -- which
    # adds exactly the white band it looks like it should remove.
    # X-LABEL PULLED UP TOWARDS THE PANELS (Leif 2026-08-29: "move the labelled molecules closer
    # to the pictures"). rect bottom and the label's y move together -- raising the label alone
    # would drop it into the legend, and lowering the rect alone just reopens the gap lower down.
    fig.tight_layout(rect=(0, 0.115, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.096, "labelled training molecules", ha="center", va="bottom",
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

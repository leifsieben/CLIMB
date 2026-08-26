"""SI Fig g — ABSOLUTE performance of the two anchors, the scaled CLIMB runs and the three
literature CLMs, one panel per task type.

ONE script, ONE figure: figures_v2/SI_fig_g.png / .pdf

WHY IT EXISTS. fig_A ranks, and a rank hides magnitude: an arm 0.001 better on forty datasets
outranks one 0.05 better on twenty-six. This plate answers the question a rank cannot -- how far
apart are they, in the units the benchmark is actually scored in (Leif 2026-08-26: "give some more
detail on how they perform absolute not just relative").

THE PREMISE IT HAD TO CORRECT. The request assumed each task category shares one error unit. That
is true of exactly one of the four:

    Activity cliffs     30 of 30 datasets are macro RMSE on pChEMBL          -> averageable
    Classification      10 roc_auc + 4 pr_auc + 1 macro_ovr_auc              -> NOT one quantity
    Virtual screening    2 nef1 + 1 roc_auc                                  -> NOT one quantity
    Regression           6 pearsonr + 4 spearmanr + 4 MAE + 3 MSE + 2 RMSE   -> NOT one quantity

Regression is the worst of them and not by a little: QM7's RMSE is ~195 kcal/mol and ESOL's is
~0.9 log units, so a mean over those two is a number about QM7 with a rounding error attached.
Averaging ROC-AUC with PR-AUC is subtler and still wrong -- PR-AUC's null value is the class
prevalence, so the two do not share a zero.

SO EACH PANEL SHOWS THE LARGEST SINGLE-METRIC SUBSET OF ITS CATEGORY, and says which one and how
many datasets in the axis label. Nothing is silently averaged across units. The panels are
therefore NOT interchangeable with fig_A's four categories -- they are the part of each category
that can honestly carry a mean, which is why this is an SI plate and not a headline.

    Activity cliffs      macro RMSE    30 of 30      the whole category
    Regression           Pearson r      6 of 19      the ADME-Fang block
    Classification       ROC-AUC         9 of 14      (MolNet:BBBP already excluded upstream)
    Virtual screening    NEF1%           2 of 3       CBS + Wong; HIV is scored on ROC-AUC

The regression panel is the thinnest cut and the one to be most careful quoting. Measured: adding
the four Spearman tasks to it moves only ECFP4 (last -> 4th) and leaves every other arm in place,
so the panel is not an artefact of which correlation metric was chosen -- but it is six ADME tasks
and the axis label says so.

ERROR BARS ARE ±1 SE ACROSS THE DATASETS IN THE PANEL, i.e. how well the panel mean is pinned
down, not how noisy one measurement is. On virtual screening that bar is enormous next to the
between-arm spread, and that is the honest reading: two datasets cannot separate this field. Do
not read the VS panel as a result.

Run:  python3 -m figures.SI_fig_g
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS, system, label as arm_label
from figures import allsuites as A
from figures import tasksuites as T

check_font()
INK = "#000000"

# ---------------------------------------------------------------- the field ---------------------
# Leif's list (2026-08-26): "only include the 2 XGBoost models and the two scaled up CLIMB models
# ... actually let's include the three literature ones as well, we have the data anyway".
#
# Deliberately NOT the fig_A field. Every arm here is either a fitted classical baseline or a
# pretrained model at 77M-plus molecules, so the plate compares like with like on scale; the 8M
# CLIMB rungs are absent because a 100M-molecule comparator beside an 8M one is a data-scale
# comparison wearing an objective comparison's clothes.
ARM_ORDER = ["ecfp", "ecfp_desc",
             "unsup_100M", "sup_dense_100M",
             "chemberta_mtr", "molformer_c3", "selfies_ted"]

# skip_dense_100M_c124 is in flight (the supervised counterpart of unsup_100M, on the same 124M
# corpus). It gets a LABELLED EMPTY BAR rather than being left out, for the same reason fig_A
# draws its pending rows: an absent arm makes the plate look finished, and this one is half the
# point of the comparison.
PENDING = {
    "sup_dense_100M": dict(system="CLIMB 100M", label="supervised, desc", color="#A3455E"),
}

# ------------------------------------------------------------- the four panels ------------------
# (category, metric, axis label, higher_is_better). The metric is DECLARED here and checked against
# the table below -- an assertion, not a comment, because "these datasets all use RMSE" is exactly
# the kind of claim that is true when written and false when read.
PANELS = [
    ("Activity cliffs",   "rmse",      "macro RMSE  (pChEMBL)",      False),
    ("Regression",        "pearsonr",  "Pearson r",                  True),
    ("Classification",    "roc_auc",   "ROC-AUC",                    True),
    ("Virtual screening", "nef1",      "NEF1%",                      True),
]


def compute():
    """Per-arm mean and SE within each panel's single-metric dataset subset."""
    have = [a for a in ARM_ORDER if a in ARMS]
    S, M = A.wide_table(have)

    # The ranking's exclusions apply here too. Drawing MolNet:BBBP's absolute AUCs on a plate that
    # sits beside a ranking it was excluded from would be two answers to one question.
    drop = sorted(set(S.columns) & T.EXCLUDED_DATASETS)
    S = S.drop(columns=drop)
    M = M.drop(index=drop)
    cat = pd.Series({c: T.category_of(c, M) for c in S.columns})

    out = {}
    for name, metric, _lab, _hb in PANELS:
        in_cat = [c for c in S.columns if cat[c] == name]
        cols = [c for c in in_cat if M.loc[c, "metric"] == metric]
        assert cols, f"SI_fig_g: no {name} dataset is scored on {metric!r}"
        # The declared metric must be the category's MOST COMMON one. If a rebuild ever makes a
        # different metric dominant, the panel silently becomes a minority cut of its category
        # and the axis label goes on saying the same thing.
        top = M.loc[in_cat, "metric"].value_counts().idxmax()
        assert top == metric, (
            f"SI_fig_g: {name} is now mostly {top!r} ({int((M.loc[in_cat,'metric']==top).sum())} "
            f"datasets) but this panel declares {metric!r} ({len(cols)}). Re-pick the panel metric "
            f"rather than leaving the label to describe a minority.")
        sub = S.loc[[a for a in ARM_ORDER if a in S.index], cols]
        n = sub.notna().sum(axis=1)
        out[name] = pd.DataFrame({
            "mean": sub.mean(axis=1),
            "se": sub.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1)),
            "n": n,
            "n_cat": len(in_cat),
            "n_used": len(cols),
        })
        # An arm on one dataset has no SE; draw the bar, drop the whisker, and mark it.
        out[name].loc[n < 2, "se"] = np.nan
    return out


def _on(hex_color):
    """Readable ink for a label drawn on top of `hex_color` (relative luminance, sRGB)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if lum > 0.45 else "white"


def _meta(a):
    if a in ARMS:
        return system(a), arm_label(a), ARMS[a]["color"]
    p = PENDING[a]
    return p["system"], p["label"], p["color"]


def main():
    res = compute()
    fig, axes = plt.subplots(2, 2, figsize=(STYLE["col2"] * 1.02, 4.35))

    for ax, (name, metric, xlab, higher) in zip(axes.ravel(), PANELS):
        d = res[name]
        for yi, a in enumerate(ARM_ORDER):
            _sys, _sub, c = _meta(a)
            if a not in d.index or not np.isfinite(d.loc[a, "mean"]):
                ax.text(0.02, yi, "in flight", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=FS["annot"] - 0.5,
                        color="#7A7A7A", style="italic", zorder=4)
                continue
            r = d.loc[a]
            ax.barh(yi, r["mean"], height=0.68, color=c, edgecolor="none", zorder=2)
            if np.isfinite(r["se"]):
                ax.errorbar(r["mean"], yi, xerr=r["se"], fmt="none", ecolor=INK,
                            elinewidth=0.9, capsize=1.8, capthick=0.8, zorder=3)
            # The number rides INSIDE the bar when it fits and outside when it does not, so a
            # short bar never pushes the axis limit out just to hold its own label.
            lab = f"{r['mean']:.3f}" + ("" if r["n"] >= 2 else " (1 ds)")
            # INK CHOSEN BY THE BAR'S LUMINANCE, not fixed to white. SELFIES-TED is #BFBFBF since
            # the literature family went grey, and white-on-#BFBFBF was unreadable -- the number
            # was still drawn, still correct, and invisible.
            #
            # AT THE BASE OF THE BAR, not at its tip: the whisker lives at the tip and a
            # right-aligned number there collided with the lower cap on every arm whose SE was
            # wide enough to matter, which is exactly the arms a reader most needs to read.
            ax.text(0.015, yi, lab, transform=ax.get_yaxis_transform(),
                    ha="left", va="center",
                    fontsize=FS["annot"] - 0.5, color=_on(c), fontweight="bold", zorder=4)

        ax.set_yticks(range(len(ARM_ORDER)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(ARM_ORDER) - 0.45, -0.55)
        arrow = "↑" if higher else "↓"
        n_used, n_cat = int(d["n_used"].iloc[0]), int(d["n_cat"].iloc[0])
        ax.set_title(f"{name} {arrow}", fontsize=FS["title"], fontweight="bold", color=INK, pad=3)
        ax.set_xlabel(f"{xlab}   ({n_used} of {n_cat} datasets)", fontsize=FS["annot"])
        ax.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=FS["tick"] - 1)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    # TWO-LINE ROW LABELS ON THE LEFT COLUMN ONLY. The bar order is identical in all four panels,
    # so repeating the names four times would cost a quarter of the plate to say the same thing.
    for ax in axes[:, 0]:
        ytrans = ax.get_yaxis_transform()
        for yi, a in enumerate(ARM_ORDER):
            _sys, _sub, _c = _meta(a)
            grey = "#7A7A7A" if a not in ARMS else INK
            st = "italic" if a not in ARMS else "normal"
            ax.text(-0.03, yi - 0.19, _sys, transform=ytrans, ha="right", va="center",
                    fontsize=FS["tick"] - 0.6, fontweight="bold", color=grey, style=st)
            ax.text(-0.03, yi + 0.20, _sub, transform=ytrans, ha="right", va="center",
                    fontsize=FS["tick"] - 1.5, color=grey, style=st)

    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    save(fig, "SI_fig_g")
    plt.close(fig)
    report(res)


def report(res):
    print("\nSI Fig g — absolute performance, single-metric subset per category\n")
    for name, metric, xlab, higher in PANELS:
        d = res[name]
        print(f"   {name}  [{metric}]  {int(d['n_used'].iloc[0])} of {int(d['n_cat'].iloc[0])} "
              f"datasets in the category")
        order = d["mean"].sort_values(ascending=not higher)
        for a, v in order.items():
            se = d.loc[a, "se"]
            bar = "" if np.isfinite(se) else "   <- one dataset, no SE"
            print(f"      {a:<18}{v:>9.4f}  +/- {se:.4f}{bar}" if np.isfinite(se)
                  else f"      {a:<18}{v:>9.4f}{bar}")
        print()


if __name__ == "__main__":
    main()

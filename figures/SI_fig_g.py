"""SI Fig g — ABSOLUTE performance of the two anchors, the scaled CLIMB runs and the three
literature CLMs, one panel per task type, ONE metric per panel.

ONE script, ONE figure: figures_v2/SI_fig_g.png / .pdf

WHY IT EXISTS. fig_A ranks, and a rank hides magnitude: an arm 0.001 better on forty datasets
outranks one 0.05 better on twenty-six. This plate answers the question a rank cannot -- how far
apart are they, in the units the benchmark is scored in (Leif 2026-08-26: "give some more detail
on how they perform absolute not just relative").

A MEAN NEEDS ONE UNIT, AND THE CATEGORIES DO NOT HAVE ONE BY DEFAULT. Each dataset is scored on
the metric its own benchmark declares, which is right for ranking and useless for averaging:
Regression alone spans pearsonr, spearmanr, MAE, MSE and RMSE, and QM7's RMSE is ~195 kcal/mol
against ESOL's ~0.9 log units. So this plate RE-READS each dataset on one metric per panel
(Leif 2026-08-26: "could we just compute NEF1% for all 3 ... for Classification could we just use
AUROC for all ... similar for regression can we find a common error unit").

That is a re-read, not a re-computation: every metric here was already computed and stored by the
runner. allsuites.wide_table(metrics=...) selects it. THE RANKING NEVER PASSES metrics -- Leif's
"don't touch the A ranking" -- so fig_A still ranks each dataset on its declared metric.

WHERE THE RE-READ CANNOT REACH, and this is a hard limit rather than a choice:

    Polaris withholds its labels. Scoring happens server-side and returns ONLY the metrics each
    benchmark declares, so a task reporting pr_auc alone cannot be re-read on ROC-AUC and one
    reporting MAE alone cannot be re-read on a correlation. Nothing local can fix that -- there is
    no y_true here to score against.

    Classification   ROC-AUC     no roc_auc for cyp2c9-substrate, cyp2d6-substrate (pr_auc only)
    Regression       Spearman    no spearmanr for caco2-wang, ld50-zhu, lipophilicity-astrazeneca,
                                 ppbr-az (MAE only), nor for MolNet ESOL/QM7 (rmse only)

Each panel says how many of its category's datasets it could re-read, and a dataset that could not
is ABSENT rather than quietly averaged in on a different metric. Activity cliffs and virtual
screening come out whole; classification and regression do not.

FartDB is macro one-vs-rest AUC over five classes -- an AUROC generalised to multiclass rather
than a different quantity -- and is counted in the ROC-AUC panel on that basis.

ERROR BARS ARE ±1 SE ACROSS THE DATASETS IN THE PANEL: how well the panel mean is pinned down, not
how noisy one measurement is. On virtual screening it is enormous next to the between-arm spread,
and that is the honest reading -- three datasets cannot separate this field. Do not read the VS
panel as a result.

Run:  python3 -m figures.SI_fig_g
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX
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
# pretrained model at 77M-plus molecules, so the plate compares like with like on data scale; the
# 8M CLIMB rungs are absent because a 100M-molecule comparator beside an 8M one is a data-scale
# comparison wearing an objective comparison's clothes.
ARM_ORDER = ["ecfp", "ecfp_desc",
             "unsup_100M", "sup_dense_100M",
             "chemberta_mtr", "molformer_c3", "selfies_ted"]

# skip_dense_100M_c124 is in flight (the supervised counterpart of unsup_100M on the same 124M
# corpus). It keeps a LABELLED GAP rather than being left out, for the same reason fig_A draws its
# pending rows: an absent arm makes the plate look finished, and this one is half the comparison.
PENDING = {
    "sup_dense_100M": dict(system="CLIMB 100M", label="supervised, desc", color="#A3455E"),
}

# ------------------------------------------------------------- the four panels ------------------
# (category, metric, axis label, higher_is_better). The metric is DECLARED here and every dataset
# in the category is re-read on it; whatever cannot supply it drops out and the count says so.
PANELS = [
    ("Activity cliffs",   "rmse",      "macro RMSE  (pChEMBL)", False),
    ("Regression",        "spearmanr", "Spearman ρ",            True),
    ("Classification",    "roc_auc",   "ROC-AUC",               True),
    ("Virtual screening", "nef1",      "NEF1%",                 True),
]

# FartDB's macro one-vs-rest AUC is an AUROC generalised to five classes, and the runner stores it
# under its own name. Mapping it into the ROC-AUC panel is a naming equivalence, declared here
# rather than assumed inside the filter, so it is visible to anyone auditing what the panel holds.
METRIC_ALIASES = {("Classification", "roc_auc"): {"macro_ovr_auc"}}


def _category_map():
    """Dataset -> task category, from the DEFAULT table (a metric override must not move it)."""
    S, M = A.wide_table([a for a in ARM_ORDER if a in ARMS])
    keep = [c for c in S.columns if c not in T.EXCLUDED_DATASETS]
    M = M.loc[keep]
    return pd.Series({c: T.category_of(c, M) for c in keep})


def compute():
    """Per-arm mean and SE within each panel, every dataset re-read on the panel's own metric."""
    have = [a for a in ARM_ORDER if a in ARMS]
    cat_default = _category_map()

    out = {}
    for name, metric, _lab, _hb in PANELS:
        in_cat = [c for c in cat_default.index if cat_default[c] == name]
        # Ask for the panel metric on EVERY dataset in the category. wide_table returns only those
        # that carry it, which is the point: absence here is a fact about what Polaris releases,
        # and it is counted rather than papered over.
        S, M = A.wide_table(have, metrics={c: metric for c in in_cat})
        ok = set(METRIC_ALIASES.get((name, metric), set())) | {metric}
        cols = [c for c in in_cat if c in S.columns and M.loc[c, "metric"] in ok]
        assert cols, f"SI_fig_g: no {name} dataset could be read on {metric!r}"

        sub = S.loc[[a for a in ARM_ORDER if a in S.index], cols]
        n = sub.notna().sum(axis=1)
        d = pd.DataFrame({"mean": sub.mean(axis=1),
                          "se": sub.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1)),
                          "n": n})
        d.loc[n < 2, "se"] = np.nan     # one dataset has no spread; draw the bar, drop the whisker
        d["n_cat"] = len(in_cat)
        d["n_used"] = len(cols)
        d.attrs["dropped"] = sorted(set(in_cat) - set(cols))
        out[name] = d
    return out


def _on(hex_color):
    """Readable ink for a label drawn on top of `hex_color` (relative luminance, sRGB)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return INK if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.45 else "white"


def _meta(a):
    if a in ARMS:
        return system(a), arm_label(a), ARMS[a]["color"]
    p = PENDING[a]
    return p["system"], p["label"], p["color"]


def main():
    res = compute()
    # 1x4 (Leif 2026-08-26). Four panels across the text block leave ~1.55in each for seven bars,
    # so the arm names cannot live on the x-axis -- they are a shared legend below, and the panels
    # keep their full width for bars instead of spending a third of it on repeated tick text.
    fig, axes = plt.subplots(1, 4, figsize=(STYLE["col2"] * 1.045, 2.78))

    for ax, (name, metric, ylab, higher) in zip(axes, PANELS):
        d = res[name]
        for xi, a in enumerate(ARM_ORDER):
            _sys, _sub, c = _meta(a)
            if a not in d.index or not np.isfinite(d.loc[a, "mean"]):
                ax.text(xi, 0.02, "in flight", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", rotation=90,
                        fontsize=FS["annot"] - 1.5, color="#7A7A7A", style="italic", zorder=4)
                continue
            r = d.loc[a]
            ax.bar(xi, r["mean"], width=0.78, color=c, edgecolor="none", zorder=2)
            if np.isfinite(r["se"]):
                ax.errorbar(xi, r["mean"], yerr=r["se"], fmt="none", ecolor=INK,
                            elinewidth=0.8, capsize=1.5, capthick=0.7, zorder=3)
            # Rotated INSIDE the bar and anchored at its base, not its tip: the whisker lives at
            # the tip, and a bar this narrow has no room beside it.
            # A BAR AVERAGED OVER FEWER DATASETS THAN ITS NEIGHBOURS IS A DIFFERENT QUANTITY,
            # and unlike a rank it cannot be rescaled to hide that. unsup_100M is still missing
            # Wong and FartDB, so its VS bar is a mean of two where the rest are means of three.
            # Marked at the bar and named in the footnote.
            short = int(r["n"]) < int(d["n_used"].iloc[0])
            ax.text(xi, 0.035, f"{r['mean']:.3f}" + (" *" if short else ""),
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", rotation=90,
                    fontsize=FS["annot"] - 1.5, color=_on(c), fontweight="bold", zorder=4)

        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.7, len(ARM_ORDER) - 0.3)
        arrow = "↑" if higher else "↓"
        n_used, n_cat = int(d["n_used"].iloc[0]), int(d["n_cat"].iloc[0])
        ax.set_title(f"{name} {arrow}", fontsize=FS["title"] - 0.5, fontweight="bold",
                     color=INK, pad=3)
        # The count is on the AXIS, not in the caption: "10 of 12" is the most misreadable thing
        # on this plate and a reader should not have to leave the panel to find it.
        ax.set_ylabel(f"{ylab}   ({n_used} of {n_cat} datasets)", fontsize=FS["annot"] - 0.5)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=FS["tick"] - 1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    handles = []
    for a in ARM_ORDER:
        sysname, sub, c = _meta(a)
        pend = a not in ARMS
        handles.append(Patch(facecolor=c, edgecolor="none", alpha=0.35 if pend else 1.0,
                             label=f"{sysname} — {sub}" + (" (in flight)" if pend else "")))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4,
               fontsize=FS["legend"] - 0.5, handlelength=1.1, handletextpad=0.45,
               columnspacing=1.1, labelspacing=0.3, borderpad=0.3, **LEGEND_BOX)
    # The short-coverage footnote is BUILT FROM THE TABLE, never typed: which arm is short of
    # which panel changes every time a rung finishes scoring, and a hand-written sentence about it
    # would be wrong within the day.
    short = {}
    for name, _m, _l, _h in PANELS:
        d = res[name]
        for a in d.index:
            if int(d.loc[a, "n"]) < int(d["n_used"].iloc[0]):
                short.setdefault(a, []).append(f"{name} {int(d.loc[a,'n'])} of "
                                               f"{int(d['n_used'].iloc[0])}")
    if short:
        txt = ";  ".join(f"{_meta(a)[0]} {_meta(a)[1]}: " + ", ".join(v)
                         for a, v in short.items())
        fig.text(0.5, 0.152, f"*  averaged over fewer datasets than the rest of the panel — {txt}",
                 ha="center", va="bottom", fontsize=FS["annot"] - 1.5, color="#4A4A4A")
    fig.tight_layout(rect=(0, 0.205, 1, 1), w_pad=1.6)
    save(fig, "SI_fig_g")
    plt.close(fig)
    report(res)


def report(res):
    print("\nSI Fig g — absolute performance, every dataset re-read on ONE metric per panel\n")
    for name, metric, ylab, higher in PANELS:
        d = res[name]
        print(f"   {name}  [{metric}]  {int(d['n_used'].iloc[0])} of {int(d['n_cat'].iloc[0])} "
              f"datasets in the category")
        if d.attrs["dropped"]:
            print(f"      cannot be re-read on {metric}: "
                  f"{', '.join(x.split(':')[1] for x in d.attrs['dropped'])}")
        for a, v in d["mean"].sort_values(ascending=not higher).items():
            se = d.loc[a, "se"]
            tail = f"  +/- {se:.4f}" if np.isfinite(se) else "   <- one dataset, no SE"
            print(f"      {a:<18}{v:>9.4f}{tail}")
        print()


if __name__ == "__main__":
    main()

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

# ------------------------------------------------------------- the six panels -------------------
# (category, metric, axis label, higher_is_better). Leif 2026-08-26: "two plots for reg and class
# then, one auroc one auprc, one mae and one spearman, but then let's keep the same definitions as
# we used for ranking in A."
#
# SO NO METRIC IS INVENTED HERE. Every metric below is one fig_A already ranks some dataset on;
# the panels differ from the ranking only in showing VALUES rather than ranks, and in grouping the
# datasets by which metric they can supply instead of mixing them.
#
# THE TWO CLASSIFICATION PANELS ARE COMPLEMENTARY, not alternative views of one set. Polaris
# releases only each benchmark's declared metric, so cyp2c9-substrate and cyp2d6-substrate can
# give pr_auc and nothing else, while the other ten give roc_auc. Ten plus two is the whole
# category -- split by what the benchmark will release, not by preference.
PANELS = [
    ("Activity cliffs",   "rmse",                "macro RMSE",            False),
    ("Regression",        "spearmanr",           "Spearman ρ",            True),
    ("Regression",        "mean_absolute_error", "MAE",                   False),
    ("Classification",    "roc_auc",             "ROC-AUC",               True),
    ("Classification",    "pr_auc",              "PR-AUC",                True),
    ("Virtual screening", "nef1",                "NEF1%",                 True),
]

# FartDB's macro one-vs-rest AUC is an AUROC generalised to five classes, and the runner stores it
# under its own name. Mapping it into the ROC-AUC panel is a naming equivalence, declared here
# rather than assumed inside the filter, so it is visible to anyone auditing what the panel holds.
METRIC_ALIASES = {("Classification", "roc_auc"): {"macro_ovr_auc"}}

# MAE IS NOT A UNIT, IT IS A UNIT PER DATASET, and this guard is why the MAE panel is trustworthy.
# Measured across the regression tasks that report it: caco2-wang 0.35 and ld50-zhu 0.63 sit beside
# ppbr-az 9.7 and the three pkis2 regressions at 18-24. That is a 69x span, so a plain mean would
# be a number about pkis2 with the other ten datasets as rounding error -- the same failure as
# averaging QM7's kcal/mol with ESOL's log units, only less obvious because both are called "MAE".
#
# A DROPPED HAND-LIST WOULD GO STALE, so the guard MEASURES instead: any dataset whose field-mean
# error exceeds SCALE_RATIO x the panel median is on a different scale and is excluded, by name, in
# the report. If a re-scoring ever moves a target's units the guard notices; a list of four task
# names would not have.
# Target axes height. Panels are ~0.65in wide once six of them share the text block, so this is
# the knob that sets their aspect; 0.80 lands close to square without squeezing the rotated in-bar
# numbers, which need ~0.30in of the column.
PANEL_H = 0.80      # axes height; panels are ~0.65in wide, so this sets the aspect
TITLE_IN = 0.22     # panel title + its pad
TOP_IN = 0.05
NOTE_IN = 0.145     # one derived footnote line
GAP_IN = 0.05       # breathing room between the lowest footnote and the axes
PAD_IN = 0.02       # under the legend box; the crop adds its own margin on top of this
WSPACE = 0.62       # enough for each panel to carry its own y tick labels
LEFT_IN = 0.44      # y tick labels + the rotated y-axis label of the first panel

SCALE_RATIO = 3.0
SCALE_GUARDED = {"mean_absolute_error", "mean_squared_error", "rmse"}


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

        offscale = []
        if metric in SCALE_GUARDED and len(cols) > 2:
            field = S[cols].mean(axis=0)                    # the field's mean error per dataset
            med = float(field.median())
            offscale = sorted(c for c in cols
                              if med > 0 and float(field[c]) / med > SCALE_RATIO)
            cols = [c for c in cols if c not in offscale]

        sub = S.loc[[a for a in ARM_ORDER if a in S.index], cols]
        n = sub.notna().sum(axis=1)
        d = pd.DataFrame({"mean": sub.mean(axis=1),
                          "se": sub.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1)),
                          "n": n})
        d.loc[n < 2, "se"] = np.nan     # one dataset has no spread; draw the bar, drop the whisker
        d["n_cat"] = len(in_cat)
        d["n_used"] = len(cols)
        d.attrs["dropped"] = sorted(set(in_cat) - set(cols) - set(offscale))
        d.attrs["offscale"] = offscale
        out[(name, metric)] = d
    return out


def _notes(res):
    """The footnote lines, BUILT FROM THE TABLE and never typed.

    Which arm is short of which panel, and which dataset sits off the MAE scale, both change as
    runs finish and as targets get re-scored. A hand-written sentence about either would be wrong
    within the day -- and the figure's own height depends on how many lines there are, so this is
    called once before the canvas is sized and once again to draw them.
    """
    lines, short = [], {}
    for name, metric, _l, _h in PANELS:
        d = res[(name, metric)]
        for a in d.index:
            if int(d.loc[a, "n"]) < int(d["n_used"].iloc[0]):
                short.setdefault(a, []).append(
                    f"{ylab_of(name, metric)} {int(d.loc[a, 'n'])}/{int(d['n_used'].iloc[0])}")
        if d.attrs["offscale"]:
            lines.append(f"{ylab_of(name, metric)} excludes "
                         f"{', '.join(x.split(':')[1] for x in d.attrs['offscale'])} "
                         f"(error >{SCALE_RATIO:g}x the panel median — a different scale, not a "
                         f"worse model)")
    if short:
        lines.insert(0, "*  averaged over fewer datasets than the rest of the panel — "
                        + ";  ".join(f"{_meta(a)[0]} {_meta(a)[1]}: " + ", ".join(v)
                                     for a, v in short.items()))
    return lines


def _note_lines(res):
    return len(_notes(res))


def _meta(a):
    if a in ARMS:
        return system(a), arm_label(a), ARMS[a]["color"]
    p = PENDING[a]
    return p["system"], p["label"], p["color"]


def main():
    res = compute()
    # ONE ROW, SIX PANELS (Leif 2026-08-26: "still have all plots be in one row"). That leaves
    # ~1.05in per panel for seven bars, so nothing that repeats per panel can afford to be text:
    # the arm names are a shared legend, the category name is carried by the title only where it
    # changes, and the in-bar numbers run vertically at the smallest size the set allows.
    # PANEL SHAPE IS SET IN INCHES, and the layout is placed rather than negotiated (Leif
    # 2026-08-26: "just don't have them be so elongated, more square like ideally"). Six panels
    # across the text block fix each at ~0.65in wide, so height is the only free variable -- and
    # the previous figure made them 0.65 x 1.77, an aspect of 2.7.
    #
    # tight_layout is NOT used here. Every piece of furniture below the axes has a fixed height in
    # inches (a two-row legend, one line per derived footnote), so the canvas is built by adding
    # those inches up and the axes are positioned from the same arithmetic. Asking tight_layout to
    # reserve a FRACTION for them meant the reserved space changed whenever the height did, which
    # is how the first attempt left an inch of white between the bars and the legend.
    # THE LAYOUT IS MEASURED, THEN PLACED (Leif 2026-08-26: "move the legend as close as
    # possible"). The legend's height is a function of its font, its row count and its longest
    # label -- none of which this module gets to decide in inches -- so the previous constant
    # reserved 0.46in for a box that renders at about 0.25in and parked a quarter-inch of white
    # under the footnotes. Draw it, ask it how tall it is, then build the canvas around the answer.
    notes = _notes(res)
    fig_w = STYLE["col2"] * 0.985
    fig = plt.figure(figsize=(fig_w, 3.0))          # provisional height, replaced below
    axes = fig.subplots(1, len(PANELS))

    prev_cat = None
    for ax, (name, metric, ylab, higher) in zip(axes, PANELS):
        d = res[(name, metric)]
        marked = False
        for xi, a in enumerate(ARM_ORDER):
            _sys, _sub, c = _meta(a)
            if a not in d.index or not np.isfinite(d.loc[a, "mean"]):
                ax.text(xi, 0.02, "in flight", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", rotation=90,
                        fontsize=FS["annot"] - 2.5, color="#7A7A7A", style="italic", zorder=4)
                continue
            r = d.loc[a]
            ax.bar(xi, r["mean"], width=0.84, color=c, edgecolor="none", zorder=2)
            if np.isfinite(r["se"]):
                ax.errorbar(xi, r["mean"], yerr=r["se"], fmt="none", ecolor=INK,
                            elinewidth=0.7, capsize=1.2, capthick=0.6, zorder=3)
            # NO VALUE PRINTED ON THE BAR (Leif 2026-08-26). At this panel size the numbers had to
            # be set vertically at 4.5pt to fit, which is smaller than the axis they duplicate --
            # the y-axis already carries the reading, and the plate exists to show the SHAPE of the
            # differences rather than to be read off digit by digit. The exact values are in the
            # module's own report() output if anyone needs them.
            #
            # THE ASTERISK STAYS, and moves above the whisker. A bar averaged over fewer datasets
            # than its neighbours is a different quantity, and unlike a rank it cannot be rescaled
            # to hide that. It is derived from the coverage table, so it vanishes on its own the
            # moment the missing cells land.
            if int(r["n"]) < int(d["n_used"].iloc[0]):
                top = r["mean"] + (r["se"] if np.isfinite(r["se"]) else 0.0)
                ax.annotate("*", (xi, top), textcoords="offset points", xytext=(0, 1.0),
                            ha="center", va="bottom", fontsize=FS["annot"], color=INK, zorder=5)
                marked = True

        # HEADROOM FOR THE ASTERISK, and only where one is drawn. Matplotlib's autoscale sizes
        # the axis to the bars and whiskers, so a mark placed above the tallest whisker lands in
        # the title. Asked for explicitly rather than by padding every panel: the five panels
        # without a short arm keep their full height for the bars.
        if marked:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.75, len(ARM_ORDER) - 0.25)
        arrow = "↑" if higher else "↓"
        n_used, n_cat = int(d["n_used"].iloc[0]), int(d["n_cat"].iloc[0])
        # EVERY panel names its category. Printing it once over a pair and leaving the second
        # with a bare arrow looked tidy and read as an orphan: a floating "down-arrow" between
        # "Regression" and "Classification" belongs to neither. The y-label carries the metric, so
        # the repetition costs nothing and the pairing stays obvious.
        head = f"{name} {arrow}"
        prev_cat = name
        ax.set_title(head, fontsize=FS["title"] - 1.5, fontweight="bold", color=INK, pad=3)
        # The count is on the AXIS, not in the caption: "10 of 12" is the most misreadable thing
        # on this plate and a reader should not have to leave the panel to find it.
        # SHORT ENOUGH TO FIT THE PANEL HEIGHT. A rotated y-label is as long as the axes are
        # tall, and at 0.80in "macro RMSE (pChEMBL) (30/30 ds)" ran off the top of the canvas --
        # drawn, correct, and cropped. The pChEMBL unit moves to the caption; the coverage count
        # stays, because it is the thing a reader most needs and least expects.
        ax.set_ylabel(f"{ylab}  ({n_used}/{n_cat})", fontsize=FS["annot"] - 1.5)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=FS["tick"] - 2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    handles = []
    for a in ARM_ORDER:
        sysname, sub, c = _meta(a)
        pend = a not in ARMS
        handles.append(Patch(facecolor=c, edgecolor="none", alpha=0.35 if pend else 1.0,
                             label=f"{sysname} — {sub}" + (" (in flight)" if pend else "")))
    # THE LEGEND, NOT THE AXES, SETS THIS PLATE'S WIDTH. save() crops to drawn content, and seven
    # keys carrying subtitles like "unsupervised, 1.1B SMILES" are wider than six 0.65in panels --
    # so shrinking the canvas moved the bars and left the crop where it was. Four columns at this
    # face fit inside the text block; a fifth, or a larger face, does not.
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
                     fontsize=FS["legend"] - 1.8, handlelength=0.95, handletextpad=0.4,
                     columnspacing=0.9, labelspacing=0.28, borderpad=0.28, **LEGEND_BOX)

    fig.canvas.draw()
    leg_in = leg.get_window_extent().height / fig.dpi
    fig_h = TOP_IN + TITLE_IN + PANEL_H + GAP_IN + NOTE_IN * len(notes) + leg_in + PAD_IN
    fig.set_size_inches(fig_w, fig_h)

    # Everything below the axes is now stacked from the canvas floor upward, in inches: legend,
    # then one line per footnote, then the axes. Each element is placed against the measured
    # height of the one under it, so a footnote appearing or disappearing -- which happens on its
    # own when unsup_100M's missing cells land -- re-flows the whole plate correctly.
    leg.set_bbox_to_anchor((0.5, PAD_IN / fig_h), transform=fig.transFigure)
    # MEASURED AGAIN AFTER THE RESIZE, and this second measurement is the one that matters. The
    # first ran on the provisional canvas and only had to be close enough to size it; using it to
    # place the footnotes put the lower line INSIDE the legend box, drawn and unreadable. Ask the
    # legend where its top actually is now.
    fig.canvas.draw()
    leg_top_in = leg.get_window_extent().y1 / fig.dpi
    for i, line in enumerate(notes):
        y_in = leg_top_in + NOTE_IN * (len(notes) - 1 - i) + 0.035
        fig.text(0.5, y_in / fig_h, line, ha="center", va="bottom",
                 fontsize=FS["annot"] - 2.0, color="#4A4A4A")
    fig.subplots_adjust(
        left=LEFT_IN / fig_w, right=1 - 0.02 / fig_w,
        bottom=(leg_top_in + NOTE_IN * len(notes) + GAP_IN) / fig_h,
        top=1 - (TOP_IN + TITLE_IN) / fig_h,
        wspace=WSPACE)
    save(fig, "SI_fig_g")
    plt.close(fig)
    report(res)


def ylab_of(name, metric):
    for n, m, lab, _h in PANELS:
        if (n, m) == (name, metric):
            return lab
    return metric


def report(res):
    print("\nSI Fig g — absolute performance, one metric per panel\n")
    for name, metric, ylab, higher in PANELS:
        d = res[(name, metric)]
        print(f"   {name} / {ylab}  [{metric}]  {int(d['n_used'].iloc[0])} of "
              f"{int(d['n_cat'].iloc[0])} datasets in the category")
        if d.attrs["dropped"]:
            print(f"      does not report {metric}: "
                  f"{', '.join(x.split(':')[1] for x in d.attrs['dropped'])}")
        if d.attrs["offscale"]:
            print(f"      OFF-SCALE, excluded (> {SCALE_RATIO:g}x panel median): "
                  f"{', '.join(x.split(':')[1] for x in d.attrs['offscale'])}")
        for a, v in d["mean"].sort_values(ascending=not higher).items():
            se = d.loc[a, "se"]
            tail = f"  +/- {se:.4f}" if np.isfinite(se) else "   <- one dataset, no SE"
            print(f"      {a:<18}{v:>9.4f}{tail}")
        print()


if __name__ == "__main__":
    main()

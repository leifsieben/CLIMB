"""Fig F — are CLIMB embeddings redundant to classical features?

ONE script, ONE figure: figures_v2/fig_F.png / .pdf  (+ figure_data/fig_F/fig_F.csv as the data record)

The test is concatenation. If the CLIMB embedding carries signal the classical features do not,
gluing it onto ECFP4+descriptors must beat ECFP4+descriptors alone. If it carries nothing new the
concatenation is at best flat — and, because the extra dimensions cost degrees of freedom, may be
slightly worse.

THE AXIS IS "% OF THE ECFP4+desc ANCHOR", ONE UNIT FOR ALL SIX PANELS (user 2026-08-19). The
panels previously each carried their own raw metric -- ROC-AUC, NEF1 and two RMSEs -- with a
per-panel reference line, so nothing could be read across panels and the compact assembly in
fig_E+F dropped the y-label entirely, leaving unlabelled bars. Now 100% is parity with the
classical anchor, by construction, and it is the dotted line in every panel. For RMSE the ratio is
inverted (100*ref/v) so that "more" still means "better" everywhere; plotting the raw ratio for an
error would make the worst arm the tallest bar in two of the six.

Read that way the result is one sentence: NO BAR REACHES 100% ON ANY CANONICAL PANEL. The CLIMB
embedding alone recovers 82-95% of the classical anchor, and adding it on top of the anchor's own
features recovers 92-99% -- i.e. concatenation does not reach the anchor, let alone beat it.

Four feature sets, same XGBoost head, same splits, same seeds:

  fp+desc       ECFP4 (2048 bits) + 217 RDKit descriptors   — the classical anchor
  CLM           the frozen CLIMB embedding alone
  desc+CLM      descriptors + CLIMB   (drops the fingerprint)
  fp+desc+CLM   everything            — the concatenation test

THE RESULT: concatenation helps on 1 of the 9 tasks run, and on NONE of the canonical panels.
`fp+desc+CLM` is worse than `fp+desc` alone on MoleculeACE (0.728 vs 0.690 macro RMSE), Ames
(-0.028), ESOL (-0.028), QM7 (-2.20), HIV (-0.018), Tox21 (-0.012) and BACE (-0.006), and EXACTLY
TIES on CBS (0.930 both). That tie is real rather than a duplicated row: NEF1 counts hits in the
top 1%, so it is quantised, and the two runs do differ on the continuous metric beside it
(ROC-AUC 0.9917 vs 0.9963).

The single exception is BBBP (+0.048 ROC-AUC, beyond its SD), and BBBP is exactly the dataset
dropped from the paper's panel set for failing to discriminate: its whole field spans 1.8% of
ROC-AUC and an UNTRAINED random encoder ranks 7 of 16 on it
(notes/bbbp-anchor-verification-2026-08-16.md). A gain on the one benchmark we already decided
cannot separate models does not rescue the conclusion. BBBP is also the only task where CLM alone
is NOT the weakest of the four feature sets — it is weakest on the other 8 of 9.

So on every benchmark that discriminates, the CLIMB embedding is redundant to the classical
featurization. This is a negative result and is reported as one. It is also the honest frame for
Fig A1, where the descriptor-bearing classical anchors rank first overall: the transformer is not
adding a missing view of the molecule.

THE NEGATIVE RESULT DOES NOT DEPEND ON THE FINGERPRINT, and that is now shown rather than hoped.
The whole test was re-run a second time on the SAME code at FP_VARIANT=ecfp4_legacy
(concat_*_legacy.csv), so legacy-vs-stereo differs by exactly one variable. Concatenation fails to
help under both featurizers on every panel that discriminates, and the fp+desc -> fp+desc+CLM gap
barely moves: Ames 0.0211 legacy against 0.0279 stereo, MoleculeACE 0.0411 against 0.0375. The
stereo fix makes the classical anchor slightly stronger and leaves the conclusion untouched.

THE ISOLATION IS PROVABLE FROM THE TABLES, WHICH IS WHY IT WAS DONE THIS WAY. `CLM` and `desc+CLM`
carry no fingerprint, so they CANNOT respond to FP_VARIANT: all 28 such rows across the two files
are identical to 8 decimal places, while 22 of the 28 fingerprint-bearing rows moved. Two arms that
must not move, and did not.

That check is also what caught the earlier mistake. An initial comparison against the 08-05/08-18
tables looked like a featurizer test and was not one -- the CLM-only rows had moved (MoleculeACE
0.840 -> 0.819, CBS NEF1 0.509 -> 0.768), which a featurizer cannot do, because three unrelated
commits had landed on the path between the runs (0ab0388, c4f1c23, 3c52686). GENERAL RULE WORTH
REUSING: any table containing an arm INVARIANT to the change under test carries its own isolation
check for free, and it costs nothing to look at it before quoting a delta.

One number not to over-read: CBS fp+desc NEF1 moves +0.0250, but NEF1 counts hits in the top 1% so
that is a single hit crossing the threshold; the ROC-AUC beside it moved +0.0010.

PANEL SCOPE: ALL SIX canonical panels are filled (2026-08-18). MoleculeACE, CBS and Ames come from
analysis/rigor/concat_panels_climb.csv; BACE, Tox21 and QM7 from the original MoleculeNet run.
Ames was the last to land, and it was never a missing RUN — the predictions had been written all
along and only the scoring failed, because that runner writes the FEATURE-SET name into the `seed`
column while the Polaris scorer calls int(seed). Its cells are 1 replicate per feature set, so they
are drawn WITHOUT a whisker rather than borrowing an SD from another panel (same rule as fig_E).
ESOL, BBBP and HIV were also run and are NOT shown here — they are outside the canonical panel set
— but they are in figure_data/fig_F/fig_F.csv and BBBP is the exception discussed above.

Error bars are +-1 SD across the seeds of that (task, feature set) cell.

Source: analysis/rigor/concat_redundancy.csv + concat_panels_climb.csv (git-tracked).

Run:  python3 -m figures.fig_F
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font
from figures.arms import PANELS, PANEL_ORDER, SHADES

check_font()
INK = "#000000"

ROOT = Path(__file__).resolve().parent.parent
# TWO sources, concatenated. `concat_redundancy.csv` is the original MoleculeNet run
# (ESOL/BBBP/BACE/HIV/Tox21/QM7); `concat_panels_climb.csv` is the canonical-panel top-up that
# added MoleculeACE and CBS (landed 2026-08-18). Same four feature sets, same XGBoost head, each
# task on its own canonical split. Ames landed 2026-08-18 in concat_panels_climb.csv: its
# predictions had been written all along and only the SCORING failed, because that runner puts the
# feature-set name in the `seed` column and the Polaris scorer does int(seed). Scored per feature
# set and appended, so the MoleculeACE and CBS rows were not clobbered.
# THE _stereo TABLES ARE THE CURRENT ONES. Both were re-run on 2026-08-19 with the current code and
# FP_VARIANT=ecfp4_stereo; the un-suffixed files are the 2026-08-05 / 08-18 vintage and are kept
# only as the provenance trail (concat_panels_climb.PREFIX_BACKUP.csv is a byte-identical copy).
#
# DO NOT PRESENT old-vs-new AS A FEATURIZER COMPARISON. It is not one, and the numbers invite the
# claim: the CLM-only arm uses no fingerprint at all, yet it moves (MoleculeACE 0.8401 -> 0.8185,
# CBS nef1 0.5089 -> 0.7678). A featurizer cannot do that. Three commits also landed on this path
# between the two runs -- 0ab0388 (standardize + median-impute classical features for non-tree
# heads), c4f1c23 (OOD prediction bounding), 3c52686 (anchor re-runs) -- so the pair confounds the
# featurizer with the code version. The defensible statement is "re-run on current code with the
# current featurizer", full stop. A clean single-variable isolation needs the same code at
# FP_VARIANT=ecfp4_legacy, where the CLM rows matching to the digit IS the check that the
# isolation worked.
SRC = ROOT / "analysis" / "rigor" / "concat_redundancy_stereo.csv"
SRC_PANELS = ROOT / "analysis" / "rigor" / "concat_panels_climb_stereo.csv"
OUTDIR = ROOT / "figures_v2"
# the per-task record is DATA, not a deliverable -- figures_v2/ holds only what goes in the paper
DATADIR = ROOT / "figure_data" / "fig_F"

# canonical panel -> task name in the source (None = experiment never run there)
PANEL_TASK = {"MoleculeACE": "MoleculeACE", "HIV": "HIV", "BACE": "BACE",
              "Ames": "Ames", "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"MoleculeACE": "macro_rmse", "HIV": "nef1", "BACE": "roc_auc",
           "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
LOWER_BETTER_METRIC = {"rmse", "macro_rmse"}
# every task in the sources, for the CSV record (superset of the canonical panels)
ALL_TASKS = {"MoleculeACE": "macro_rmse", "CBS": "nef1", "ESOL": "rmse", "QM7": "rmse",
             "BACE": "roc_auc", "BBBP": "roc_auc", "Ames": "roc_auc", "Tox21": "roc_auc",
             "HIV": "roc_auc"}
# classical anchor keeps the anchor amber; anything containing CLIMB moves into the unsup blues,
# darkening as more classical information is added back
# THE FULL LATTICE (user 2026-08-19). Every combination of the two classical blocks with at most
# one embedding: 4 classical bases x {no embedding, + CLIMB, + CheMeleon}, minus the empty cell.
# CLIMB+CheMeleon is deliberately absent -- stacking two learned embeddings answers no question
# this paper asks.
#
# COLOUR IS THE EMBEDDING, GROUP IS THE CLASSICAL BASE, and that is what makes the panel answer
# both directions of the redundancy question at once:
#   read WITHIN a group   -> does the embedding add anything to this classical block?
#   read ONE COLOUR ACROSS groups -> does the classical block add anything to this embedding?
# Redundancy is directional, so a figure that only shows the first reading cannot support the
# claim: a flat within-group result is equally consistent with the head failing to exploit 512
# dense dims beside 2048 sparse bits, with added variance at fixed n, or with a task ceiling.
#
# The desc+ECFP4 group is also the POSITIVE CONTROL -- "does adding a second block ever help on
# this task at all" -- which is what turns a flat embedding result from unreadable into negative.
# It was a dotted reference line for one revision; back to bars so it sits on the same footing as
# everything else it is being compared against.
NO_EMB, EMB_CLM, EMB_CHE = "no embedding", "+ CLIMB", "+ CheMeleon"
ROLE_COLOR = {NO_EMB: SHADES["anchor"][0], EMB_CLM: SHADES["unsup"][0],
              EMB_CHE: SHADES["chemeleon"][0]}
GROUPS = [("none", None,
           [(None, NO_EMB), ("CLM", EMB_CLM), ("CheMel", EMB_CHE)]),
          ("desc", "desc",
           [("desc", NO_EMB), ("desc+CLM", EMB_CLM), ("desc+CheMel", EMB_CHE)]),
          ("ECFP4", "fp",
           [("fp", NO_EMB), ("fp+CLM", EMB_CLM), ("fp+CheMel", EMB_CHE)]),
          ("desc+ECFP4", "fp+desc",
           [("fp+desc", NO_EMB), ("fp+desc+CLM", EMB_CLM), ("fp+desc+CheMel", EMB_CHE)])]
FEATURES = [(k, role, ROLE_COLOR[role]) for _, _, mem in GROUPS for k, role in mem if k]
BASE, CONCAT = "fp+desc", "fp+desc+CLM"

# ---------------------------------------------------------------------------------------------
# THE AXIS: percent of the classical anchor, so all six panels share one unit
# ---------------------------------------------------------------------------------------------
# Raw metrics cannot share an axis -- ROC-AUC, NEF1 and two RMSEs in different units -- so every
# panel used to carry its own scale and its own dotted reference line, and nothing could be read
# across panels. Expressing each feature set as a PERCENTAGE OF THE ECFP4+desc ANCHOR puts them
# on one axis: 100% is parity with the anchor, by construction, and is the dotted line.
#
# For higher-is-better metrics that is 100*v/ref. For RMSE it is 100*ref/v -- the RATIO IS
# INVERTED so that "more" still means "better" on the same axis; plotting 100*v/ref for an error
# would make the worst arm the tallest bar in two of the six panels.
#
# The anchor's OWN uncertainty is deliberately not propagated: it is the reference, it is 100% by
# definition, and folding its spread into every other bar would make the comparison to it noisier
# than it is. Each bar's error is its own spread rescaled by the same factor as its mean, so the
# error bars answer "how well is this feature set measured", not "is it distinguishable from the
# anchor" -- that second question is what the delta column of fig_F.csv is for.
PCT_YLABEL = "% of ECFP4+desc"
PCT_YLIM = (0, 132)


def as_pct_of_anchor(values, errs, metric):
    """(values, errs) rescaled so the anchor is 100. Returns NaNs unchanged."""
    ref = values[0]
    if not np.isfinite(ref) or ref == 0:
        return [np.nan] * len(values), [0.0] * len(values)
    lower_better = metric in LOWER_BETTER_METRIC
    out_v, out_e = [], []
    for v, e in zip(values, errs):
        if not np.isfinite(v) or v == 0:
            out_v.append(np.nan); out_e.append(0.0); continue
        if lower_better:
            out_v.append(100.0 * ref / v)
            out_e.append(100.0 * ref * e / (v * v))      # |d(ref/v)/dv| * e
        else:
            out_v.append(100.0 * v / ref)
            out_e.append(100.0 * e / ref)
    return out_v, out_e


def compute():
    """The concatenation table. Exposed so figures/fig_E_plus_F.py can assemble this figure with fig_E
    without re-implementing (and therefore drifting from) the analysis."""
    return pd.concat([pd.read_csv(SRC), pd.read_csv(SRC_PANELS)], ignore_index=True)


# Panels that share a METRIC share a y-RANGE, so a bar of a given height means the same thing in
# each (user 2026-08-19: "some go to 1 others don't, seems inconsistent"). Panels on different
# metrics cannot share one -- macro RMSE, NEF1% and kcal/mol are not commensurable -- so the rule is
# per metric, not global. Computed from the drawn data rather than hardcoded.
def shared_ylims(d):
    """{metric: (lo, hi)} over every canonical panel scored on that metric."""
    import collections
    acc = collections.defaultdict(list)
    for p in PANEL_ORDER:
        task = PANEL_TASK.get(p)
        if task is None:
            continue
        metric = PRIMARY[task]
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        for f, _, _ in FEATURES:
            if f in g.index:
                v, e = float(g.loc[f, "mean"]), float(g.loc[f, "std"])
                e = 0.0 if e != e else e
                acc[metric] += [v - e, v + e]
    out = {}
    for metric, vs in acc.items():
        lo, hi = min(vs), max(vs)
        pad = 0.10 * max(hi - lo, 1e-9)
        out[metric] = (lo - pad, min(hi + pad, 1.0) if metric == "roc_auc" else hi + pad)
    return out


def draw_panel(ax, d, p, compact=False, tag=None, fig=None, ylims=None, xrot=None, bw=None):
    """Draw ONE canonical panel onto an existing axes.

    `compact` narrows the bars and drops the y-label, for the assembled fig_E+F where six panels
    share half the canvas. `tag` puts the panel letter ON THE TITLE BASELINE, immediately left of
    the title, matching fig_E's tag/title pair -- so every panel letter in the assembled figure is
    positioned the same way instead of F's floating above its centred title.
    """
    meta = PANELS[p]
    task = PANEL_TASK[p]
    arrow = "↑" if meta["higher_better"] else "↓"
    if tag is None:
        ax.set_title(f"{meta['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=3)
    else:
        from matplotlib.transforms import ScaledTranslation
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        # short label in the assembled figure: "Ames Mutagenicity" runs into the next panel's tag
        # short names in the assembled figure: the full labels run into the next panel's tag
        short = {"Ames": "Ames", "MoleculeACE": "MolACE"}.get(p, meta["label"])
        ax.text(0.0, 1.04, f"{short} {arrow}", fontsize=FS["title"] - (1 if compact else 0),
                fontweight="bold",
                va="bottom", ha="left", color=INK,
                transform=ax.transAxes + ScaledTranslation(11 / 72, 0, fig.dpi_scale_trans))
    # ABSOLUTE metric (user 2026-08-19: "I do think absolute performance was more meaningful").
    # The % -of-anchor axis was tried and dropped: it made the panels comparable to each other but
    # answered the wrong question, since the comparison that matters here is base vs base+embedding
    # WITHIN a group, and that is a like-for-like pair already. Labelled in every panel including
    # the compact assembly, where it used to be dropped and left the bars unlabelled.
    ax.set_ylabel(meta["metric_short"], fontsize=FS["annot"] - (1 if compact else 0), color=INK)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if task is None:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        return
    metric = PRIMARY[task]
    g = d[(d.task == task) & (d.metric == metric)].set_index("features")

    def cell(k):
        if k not in g.index:
            return None, None
        v = float(g.loc[k, "mean"])
        e = float(g.loc[k, "std"])
        return v, (0.0 if not np.isfinite(e) else e)

    # x positions: two groups of three with a gap between, so the eye reads "base, +CLIMB,
    # +CheMeleon" twice rather than six unrelated bars.
    GAP = 0.8
    xs, ys, es, cs, present = [], [], [], [], []
    xi = 0.0
    group_centres = []
    for _, _, members in GROUPS:
        start = xi
        for k, role in members:
            if k is None:
                # "no classical, no embedding" is the empty feature set -- it holds a slot so the
                # three colours stay in the same order in every group, and draws nothing.
                xi += 1.0
                continue
            v, e = cell(k)
            xs.append(xi); ys.append(v); es.append(e)
            cs.append(ROLE_COLOR[role]); present.append(v is not None)
            xi += 1.0
        group_centres.append((start + xi - 1.0) / 2.0)
        xi += GAP
    x = np.array(xs)

    base_v = None      # desc+ECFP4 is a BAR again, not a reference line
    drawn = [v for v in ys if v is not None]
    if not drawn and base_v is None:
        ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS["annot"], color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        return

    # y range first: the placeholders need a height to sit in, and the range must not depend on
    # which arms happen to have landed -- otherwise the axis moves as data arrives and two
    # printings of "the same" figure disagree.
    # Include the ERROR BARS, not just the means: sizing to the means alone let MoleculeACE's
    # whisker run off the top of the panel.
    span_vals = [v + e for v, e in zip(ys, es) if v is not None]
    span_vals += [v - e for v, e in zip(ys, es) if v is not None]
    span_vals += [base_v] if base_v is not None else []
    lo, hi = min(span_vals), max(span_vals)
    pad = 0.42 * max(hi - lo, 1e-9)
    y0, y1 = lo - pad, hi + pad
    if meta["metric"] == "roc_auc":
        y1 = min(y1, 1.0)
    ax.set_ylim(y0, y1)

    bw = bw if bw is not None else (0.62 if compact else 0.74)
    for xi_, v, e, c, ok in zip(x, ys, es, cs, present):
        if ok:
            ax.bar([xi_], [v], width=bw, color=c, edgecolor=INK, linewidth=0.7,
                   yerr=[e], error_kw=dict(elinewidth=0.9, capsize=1.8, capthick=0.9,
                                           ecolor=INK, zorder=6), zorder=3)
        else:
            # NOT RUN, drawn as an empty slot rather than omitted. A missing bar and a bar at the
            # axis floor look identical, and here the two mean opposite things -- "no measurement"
            # against "the embedding destroyed the score".
            ax.bar([xi_], [y1 - y0], bottom=y0, width=bw, facecolor="none",
                   edgecolor=STYLE["grid"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
            ax.text(xi_, y0 + 0.5 * (y1 - y0), "not run", rotation=90, ha="center", va="center",
                    fontsize=FS["annot"] - 2, color="#8A8A8A", zorder=4)

    # NO per-bar tick labels. Eleven of them do not fit at any panel width in this set, and they
    # would repeat what the colour already says -- the legend names the three embeddings once and
    # the group name names the classical base. Ticks are the group names alone.
    ax.set_xticks(group_centres)
    # Rotated in BOTH layouts: "desc+ECFP4" beside "ECFP4" needs ~0.9in horizontally and the
    # widest panel in this set gives ~0.55in per group.
    ax.set_xticklabels([g for g, _, _ in GROUPS], fontsize=FS["annot"] - 1, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    if compact:
        ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))


# Wrapped forms for the VERTICAL legend. "CLIMB + descriptors + ECFP4" on one line is 3.3in wide,
# which forces the legend into a horizontal strip; broken at the + signs it is 3 short lines and
# the whole key becomes a tall narrow block (user 2026-08-19).
WRAPPED = {"CLIMB alone": "CLIMB\nalone",
           "CLIMB + descriptors": "CLIMB\n+ descriptors",
           "CLIMB + descriptors + ECFP4": "CLIMB\n+ descriptors\n+ ECFP4"}


def legend_handles(skip_anchor=False, wrap=False):
    """Three ROLES, not six bars, plus the dotted baseline.

    The same three colours repeat in both groups -- classical block alone, + CLIMB, + CheMeleon --
    so naming them once is the whole key; which classical block a group uses is written under its
    own bars. Six entries would just be the same three words twice.

    The dotted line gets its OWN entry (user 2026-08-19). It used to be the anchor's bar as well,
    which was redundant; with the bar gone the line is the only thing carrying fp+desc, and an
    unlabelled reference line is a number the reader cannot name.

    `skip_anchor` and `wrap` are kept for the fig_E+F caller's signature; the role legend is short
    enough that neither changes it now.
    """
    return [Patch(facecolor=ROLE_COLOR[r], edgecolor=INK, lw=0.8, label=r)
            for r in (NO_EMB, EMB_CLM, EMB_CHE)]


def main():
    d = compute()

    # ---- data record: every task the experiment covers, not just the canonical panels ----
    rows = []
    for task, metric in ALL_TASKS.items():
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        if BASE not in g.index or CONCAT not in g.index:
            continue
        sign = -1 if metric in LOWER_BETTER_METRIC else 1
        delta = sign * (float(g.loc[CONCAT, "mean"]) - float(g.loc[BASE, "mean"]))
        sd = float(g.loc[CONCAT, "std"])
        sd = 0.0 if not np.isfinite(sd) else sd     # 1-replicate cell: no SD to beat
        row = dict(task=task, metric=metric,
                   in_canonical_panels=int(task in {v for v in PANEL_TASK.values() if v}))
        for f, _, _ in FEATURES:
            row[f] = round(float(g.loc[f, "mean"]), 4) if f in g.index else ""
            _sd = float(g.loc[f, "std"]) if f in g.index else float("nan")
            row[f + "_sd"] = round(_sd, 4) if np.isfinite(_sd) else ""
        row.update(delta_vs_fp_desc=round(delta, 4), concat_sd=round(sd, 4),
                   beats_sd="yes" if delta > sd else "no")
        rows.append(row)
    OUTDIR.mkdir(exist_ok=True)
    cols = ["task", "metric", "in_canonical_panels"] + \
           [c for f, _, _ in FEATURES for c in (f, f + "_sd")] + \
           ["delta_vs_fp_desc", "concat_sd", "beats_sd"]
    DATADIR.mkdir(parents=True, exist_ok=True)
    with open(DATADIR / "fig_F.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- the figure: canonical six ----
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 5.1))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        draw_panel(ax, d, p)

    handles = legend_handles()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=4,
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
               borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    # COMPONENT of fig_E+F, so it belongs in panels/ with fig_C1/C2/D and fig_A1/A2 --
    # figures_v2/ proper should hold only what goes in the paper. It is still rendered
    # standalone for review.
    save(fig, "fig_F", subdir="panels")
    plt.close(fig)

    print("\nFig F — does concatenating CLIMB onto the classical features help?")
    print("  (delta signed so + = concatenation helped)\n")
    print(f"  {'task':<13}{'canon':<7}" + "".join(f"{f:>16}" for f, _, _ in FEATURES) +
          f"{'delta':>10}{'> SD?':>7}")
    for r in rows:
        line = f"  {r['task']:<13}{'yes' if r['in_canonical_panels'] else '—':<7}"
        for f, _, _ in FEATURES:
            v = r.get(f)
            line += (f"{v:>16.4f}" if isinstance(v, (int, float)) else f"{'—':>16}")
        line += f"{r['delta_vs_fp_desc']:>+10.4f}{r['beats_sd']:>7}"
        print(line)
    helped = sum(r["beats_sd"] == "yes" for r in rows)
    print(f"\n  concatenation beat its own SD on {helped}/{len(rows)} tasks")
    print("  wrote figures_v2/fig_F.png/pdf + figure_data/fig_F/fig_F.csv")


if __name__ == "__main__":
    main()

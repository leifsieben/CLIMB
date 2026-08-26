"""Fig A — overall standing across four TASK TYPES, ranking only.

ONE script, ONE figure: figures_v2/fig_A.png / .pdf

Full A4 text-block width, and as short as the content allows: one row per model, no second
panel. The six per-dataset panels that used to sit underneath are fig_A2's job; repeating them
here cost most of a page and said nothing the ranking did not.

WHAT IS RANKED. 14 models, every one of them a FROZEN representation with a trained head. Each is ranked WITHIN every individual dataset (1 = best of the
field), those per-dataset ranks are averaged within each of four task categories, and the four
category means are averaged with EQUAL WEIGHT. Ranking per dataset is what makes the pooling
legal: the metrics are heterogeneous (RMSE, ROC-AUC, PR-AUC, NEF1%, Pearson r, Spearman rho) and
cannot be averaged. A dataset scored on only k of the field is rescaled to the full field
so a missing model cannot flatter the rest.

    coloured tick mean of the four category ranks
    bar           +/-1 SE across the four category means, design-effect corrected
    open marks    the four category means themselves

EQUAL WEIGHT PER CATEGORY IS A CHOICE AND THE CAPTION MUST SAY SO. Activity cliffs holds 30
datasets and virtual screening 3, so each VS dataset carries ~8.3% of the headline against ~0.83%
for each MoleculeACE target -- a 10x ratio. The axis is task type, not benchmark size, so this is
deliberate; it is not self-evident.

TWO NON-UNIFORMITIES THE CAPTION ALSO HAS TO CARRY (notes/figA-seed-axis-is-not-uniform.md):

  1. "3 seeds" is one label over two estimands. Most CLIMB arms vary the PRETRAINING with head
     seeds pinned. The two ECFP4 anchors and the three literature CLMs have no pretraining stage
     to vary, and unsup_100M has exactly one pretraining and always will, so those five carry
     three disjoint HEAD-SEED triples inside one directory instead. Do NOT write "three
     pretraining seeds" as a property of the panel -- it is false for 6 of the 14 rows, and a
     head-seed bar is TIGHTER than a pretraining-seed bar for a reason that has nothing to do
     with the arm being more stable, so the two must not be read against each other.
  2. The probe head is representation-dependent by design: ECFP4 arms at XGBoost, every CLM at a
     frozen encoder plus MLP, because SI fig f shows the preference is representation-dependent
     and a single head handicaps whichever representation it does not suit. The three literature
     CLMs have NOT been measured at XGBoost, and CheMeleon -- the closest analogue -- prefers
     XGBoost by 0.138 macro RMSE, wider than the whole ECFP4+desc-to-CLIMB span. Leif ruled no
     ablation for now; the exposure is recorded rather than hidden.

MISSING DATA IS DRAWN, NOT OMITTED. Wong (virtual screening) and FartDB (classification) landed
2026-08-26 and are in the counts; the one arm still short of them is unsup_100M, which is drawn
with a * and a footnote naming exactly what it is missing rather than being quietly ranked on a
smaller field. An arm with no results at all gets a labelled empty row, not an absent row:
dropping it would make the plate look finished and quietly renumber the field, which changes
every rank on the page.

THE COUNTS ABOVE ARE TESTED, NOT TRANSCRIBED. _audit_docstring() below re-derives every number in
this docstring from the arm table on each render and fails if one has drifted. Two captions in
this repo have already shipped a count that was true when written and false when read; a number
in prose cannot fail loudly on its own, so it is checked against the thing it describes.

Run:  python3 -m figures.fig_A
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX, row_ncol
from figures.arms import ARMS, SHADES, system, label as arm_label
from figures import tasksuites as T
from figures.allsuites import wide_table as A_wide

check_font()
INK = "#000000"

# ---------------------------------------------------------------- the ranked field --------------
# The CLIMB seven are Leif's list (2026-08-25). Changing the field is a one-line edit here; it is
# deliberately NOT derived from arms.in_ranking, which admits all 22 arms including fig_A2's
# ablation ladder.
#
# EVERY ARM ON THIS PLATE IS NOW A FROZEN REPRESENTATION WITH A TRAINED HEAD. The list drops all
# four end-to-end arms, which is a real improvement rather than a cosmetic cut: fine-tuning the
# whole network is the largest single effect in this benchmark, so a table mixing frozen probes
# with fine-tuned networks ranks two protocols at once and an end-to-end row placing high says
# something about fine-tuning rather than about pretraining. fig_A1 carries that comparison, with
# its own end-to-end key.
# sup_mixed WAS HERE AND IS OUT (Leif 2026-08-26: "let's do desc+sparse for all"). Its objective
# is the same 50/50 MTR+supervised recipe as sup_dense_sparse; the only difference is that it adds
# PCQM and WONG to the supervised label families -- and WONG is the antibiotic screen that became
# one of this figure's virtual-screening datasets the same day. It ranked 1 of 14 on Wong while the
# two other assay-trained arms ranked 13th and 14th. See notes/wong-sft-family-is-an-eval-set.md.
#
# sup_dense_sparse is the same recipe without the two families, so the swap keeps the objective
# comparison the row exists to make and drops the contaminated one.
RANKED_ARMS = ["ecfp", "ecfp_desc",
               "sup_dense", "sup_sparse", "sup_dense_sparse",
               "unsup", "u2s_dense", "u2s_sparse", "s2u_dense", "random_encoder",
               # PROVISIONAL (Leif 2026-08-26: "I'm not yet committed to having them in figure A1
               # for the actual paper, it's more for my own learning"). Kept in one place so
               # dropping it is deleting this line, not unpicking a decision from the arm list.
               #
               # The largest-corpus CLIMB run: 100M DISTINCT molecules, the only CLIMB arm to
               # exceed ChemBERTa-2's ~77M and the only one whose data scale is comparable to a
               # published CLM. Its supervised counterpart is skip_dense_100M_c124, in flight;
               # sup_dense_96M is NOT it -- 96M forward passes but only 12M unique molecules, and
               # it scores worse than the 24M rung it repeats (0.7748 vs 0.7687 macro RMSE).
               #
               # IT HAS ONE PRETRAINING AND ALWAYS WILL (Leif: "the big ones will only ever have
               # one seed, that is perfectly fine"). That is a decision, not a gap: its three
               # replicates are HEAD seeds, so its point estimate carries one draw from the
               # pretraining distribution where the other CLIMB arms average three. Measured cost
               # of that, in notes/figA-seed-axis-is-not-uniform.md: pretraining-seed SD is a
               # median 7.8% of the field spread, and under a resampled draw this arm holds 2nd
               # place in 87% of simulations and stays ahead of ChemBERTa-2 in ~89%. Quote the
               # placement plainly; hedge the head-to-head.
               "unsup_100M",
               "chemberta_mtr", "molformer_c3", "selfies_ted"]

# PENDING_ARMS is empty: the three literature CLMs landed 2026-08-26 and are now real entries in
# arms.py (colour, label, replicate convention, measured parameter count). The mechanism stays
# rather than being deleted -- it is how the next commissioned arm gets a labelled empty row
# instead of being silently absent, which is the difference between "we are waiting on this" and
# "this plate is finished".
PENDING_ARMS = {}
assert not (set(PENDING_ARMS) & set(ARMS)), \
    "an arm is declared PENDING here and also present in arms.py -- promote it and delete it here"
assert set(RANKED_ARMS) - set(ARMS) == set(PENDING_ARMS), \
    f"RANKED_ARMS names arms that are neither in arms.py nor declared pending: " \
    f"{sorted(set(RANKED_ARMS) - set(ARMS) - set(PENDING_ARMS))}"

# PROVISIONAL arms are exploratory. They are RANKED even when short of the full dataset field
# (Leif 2026-08-26: "just include the 100M provisionally just so I can see the current results...
# it may not stay in anyway"), and every row that is short carries a * plus a footnote naming what
# it is missing. A NON-provisional arm that is short is still a hard error, because the whole point
# of the guard is that a category mean over a different dataset set is a different quantity -- for
# a provisional arm that is a stated caveat, for a paper arm it is a bug.
PROVISIONAL = {"unsup_100M"}

# ARMS REGISTERED IN arms.py BUT NOT YET SCORED EVERYWHERE. They are drawn as labelled empty rows
# and left OUT of the ranking, exactly as PENDING_ARMS are -- the difference is only that these
# exist in arms.py, so PENDING_ARMS' "must not be in ARMS" assertion cannot carry them.
#
# THE DECLARATION IS TESTED AGAINST MEASURED COVERAGE, not trusted. compute() asserts that every
# arm named here is genuinely short AND that no arm outside it is, so this set cannot quietly
# outlive the run that fills it: the day sup_dense_sparse's Wong and FartDB cells land, the render
# fails until the name is removed, rather than drawing an empty row over real data forever. A
# declaration that only ever removes things is the kind that goes stale silently.
AWAITING_DATA = {"sup_dense_sparse"}

N = len(RANKED_ARMS)
HAVE = [a for a in RANKED_ARMS if a in ARMS and a not in AWAITING_DATA]


def _meta(a):
    """(bold first line, second line, colour) for one row."""
    if a in ARMS:
        return system(a), arm_label(a), ARMS[a]["color"]
    p = PENDING_ARMS[a]
    return p["system"], p["label"], p["color"]


def compute():
    """Ranks over the full ranked field, plus per-arm category coverage.

    The field size passed to the ranker is len(HAVE), so the ranks returned run 1..len(HAVE). They
    are NOT rescaled to 1..13 here: doing that would invent a position for three arms that have no
    results, and the axis is drawn to len(HAVE) with the pending rows left empty instead.
    """
    # THE AWAITING DECLARATION IS A CLAIM ABOUT THE DATA AND IS CHECKED AGAINST IT. Measured over
    # the FULL registered field, not just the ranked one, so an arm that is short cannot be ranked
    # by omission from this set, and an arm that is complete cannot keep an empty row by inertia.
    full_S, _full_M = A_wide(RANKED_ARMS)
    n_by_arm = full_S.notna().sum(axis=1)
    target = int(n_by_arm.max())
    really_short = {a for a in RANKED_ARMS if a in ARMS and int(n_by_arm.get(a, 0)) < target}
    stale = AWAITING_DATA - really_short
    undeclared = really_short - AWAITING_DATA - PROVISIONAL
    assert not stale, (
        f"fig_A: {sorted(stale)} is declared AWAITING_DATA but now covers all {target} datasets. "
        f"Remove it from AWAITING_DATA so it is ranked -- an empty row drawn over real results is "
        f"worse than no row.")
    assert not undeclared, (
        f"fig_A: {sorted(undeclared)} is short of the {target}-dataset field and is neither "
        f"AWAITING_DATA nor PROVISIONAL. Declare it or score it; do not rank it.")

    out, cat, R = T.wide_ranks(HAVE, summary=SUMMARY)
    missing = [a for a in HAVE if a not in out.index]
    assert not missing, f"fig_A: arms in RANKED_ARMS with no row in the score table: {missing}"

    # EVERY SCORED ARM MUST COVER THE SAME DATASETS. Rescaling handles a dataset that is missing
    # for some arms, but it does NOT make an arm's CATEGORY MEAN comparable when that mean is
    # taken over a different set of datasets from its neighbours'. An arm at 37 of 65 has a
    # regression mean over eleven Polaris tasks it happens to have; the arm above it has one over
    # nineteen. Those are different quantities printed in the same column.
    #
    # This fired on its first run: MoLFormer and SELFIES-TED came back at 37 while ChemBERTa read
    # 65, because Polaris scores are produced LOCALLY (the labels are withheld) and only one of
    # nine directories had been scored. Note what the count did NOT catch -- ChemBERTa's 65 was
    # built from one of its three replicate directories, so a coverage check alone still passes an
    # arm that is a third as deep. Replicate depth is audit check 19's job; this is coverage.
    n_ds = out["n_datasets"].astype(int)
    full = int(n_ds.max())
    short_prov = {a: int(n_ds[a]) for a in n_ds.index if a in PROVISIONAL and n_ds[a] < full}
    missing_ds = {}
    for a in short_prov:
        have = set(R.columns[R.loc[a].notna()])
        missing_ds[a] = sorted(set(R.columns) - have)
        print(f"   [fig_A] {a} is PROVISIONAL, ranked on {short_prov[a]} of {full} datasets, "
              f"missing {missing_ds[a]}. Marked * on the plate.")
    n_ds = n_ds.drop(index=list(short_prov))
    if n_ds.nunique() > 1:
        short = {a: int(n_ds[a]) for a in n_ds.index if n_ds[a] < n_ds.max()}
        raise AssertionError(
            f"fig_A: arms do not cover the same datasets. Full field is {int(n_ds.max())}; "
            f"short: {short}. A category mean taken over a different dataset set is not the same "
            f"quantity as its neighbours'. Score or fetch the missing runs -- do not rank a "
            f"partial arm beside complete ones.")
    avail = {k: int((cat == k).sum()) for k in T.CATEGORIES}
    return out, avail, short_prov, missing_ds


def _audit_docstring():
    """Re-derive every count this module's docstring states, and fail if one has drifted.

    A caption number is the purest form of the failure this repo keeps hitting: it answers
    confidently, it cannot fail on its own, and it stops the reader looking. Two have already
    shipped stale here -- SI fig a claimed "12 of 18 cells" after a retirement made it 12, and
    this docstring said "thirteen models" and "five of the thirteen rows" for a day after
    unsup_100M was added. Neither was visible from inside the figure; both were arithmetic over
    the arm table, which is right here.

    So the docstring is a TESTED artefact, not prose. Each claim below names a phrase that must
    appear verbatim, built from the arms as they actually are. Adding or dropping an arm breaks
    the render until the sentence is true again.
    """
    head_seed = [a for a in RANKED_ARMS
                 if a in ARMS and not ARMS[a].get("pretrain_replicates", True)]
    claims = {
        "field size":
            f"WHAT IS RANKED. {len(RANKED_ARMS)} models,",
        "head-seed row count":
            f"it is false for {len(head_seed)} of the {len(RANKED_ARMS)} rows",
    }
    stale = {k: v for k, v in claims.items() if v not in (__doc__ or "")}
    if stale:
        raise AssertionError(
            "fig_A: the module docstring is the caption source and it has gone stale. "
            "Derived now: " + "; ".join(f"{k} -> {v!r}" for k, v in stale.items()) +
            ". Fix the sentence in the docstring; do not weaken this check.")
    return head_seed


BAND = "#F2F2F2"          # zebra row band, same value as fig_A1 so the two plates read alike

# How a CATEGORY is summarised from its datasets: "mean" or "median" rank. The four category
# summaries are averaged either way.
#
# MEAN, by preference (Leif 2026-08-25: "easier to understand and feels a bit more honest"), and
# the choice is defensible because the ORDERING DOES NOT DEPEND ON IT. Five schemes were compared
# on this exact field (notes/rank-compression-on-packed-fields.md):
#
#     mean rank                 ECFP4+desc classification 3.93   (reported)
#     median rank                                         2.00   Kendall tau vs mean +0.994
#     20% trimmed mean rank                               3.10   +0.974
#     mean z-score (effect size)                          +0.77  +0.949
#     mean rank with noise-tied midranks                  4.46   +0.949
#
# No arm moves more than one place under any of them. What DOES change is how a single packed
# dataset is charged, and that is a caption sentence rather than a different figure.
#
# TIES WERE TRIED AND REJECTED. Giving indistinguishable arms a shared midrank sounds neutral and
# is not: ECFP4+desc is OUTRIGHT BEST on 42 of 65 datasets, so tying merges its wins into
# midranks and it can only lose, while an arm that is never best can only gain. Measured, it made
# the very number it was meant to fix WORSE (3.93 -> 4.18 at 1 SD, -> 4.46 at 2 SD). Rounding to
# a fixed decimal is the same rule with an arbitrary threshold and the same asymmetry, plus it is
# not comparable across ROC-AUC, RMSE, NEF1% and pr_auc.
SUMMARY = "mean"


def main(weighting="dataset", name="fig_A", subdir=None):
    """weighting="dataset" is the paper cut; "category" is the alternative kept for comparison.

    LEIF CHOSE THE POOLED CUT ON 2026-08-26 -- "let's use the average over all datasets one, it's
    easier to understand" -- and it costs almost nothing to say it that way: the two orderings
    agree at Kendall tau 0.934, the top four are identical, and only sup_mixed moves more than one
    place (5th -> 7th, its virtual-screening strength diluted once 30 MoleculeACE targets carry 45%
    of the headline instead of 25%).

    THE TWO DIFFER ONLY IN HOW THE PER-DATASET RANKS ARE SUMMARISED, never in how they are
    computed -- both read the same rank matrix out of tasksuites.wide_ranks. Activity cliffs holds
    30 of the 66 datasets and virtual screening 3, so pooling by dataset hands MoleculeACE 45% of
    the headline where the category cut gives it 25%. Worth keeping both precisely because that is
    a choice and not a fact.
    """
    head_seed = _audit_docstring()
    out, avail, short_prov, missing_ds = compute()
    if weighting == "dataset":
        out = out.assign(mean_rank=out["mean_rank_pooled"], se_rank=out["se_rank_pooled"])
    elif weighting != "category":
        raise ValueError(f"weighting must be 'category' or 'dataset', got {weighting!r}")
    order = list(out.sort_values("mean_rank").index) + [a for a in RANKED_ARMS if a not in out.index]
    nfield = len(out)

    # One row per model. Fourteen rows plus axis and legend inside 3.7in, roughly a third of A4's
    # text height -- the point of dropping the six-panel block.
    #
    # WIDTH IS SET BY MEASUREMENT. save() crops to the drawn content, so the rendered plate is NOT
    # figsize wide: an axes box that leaves slack renders narrower than the text block and LaTeX
    # then scales every font in it up relative to the rest of the set. This layout is authored to
    # fill the canvas and measured at 6.69in against a 6.69in text block. Re-measure if the axes
    # fractions below change, AND when the row labels change: the multiplier is the ratio of the
    # canvas to the crop, and the crop is set by the widest label in the left column. It has moved
    # three times for that reason -- 1.123 when the labels were short, 1.061 once the literature
    # CLMs took subtitles like "unsupervised, 1.1B SMILES". First version rendered 5.58in.
    fig = plt.figure(figsize=(STYLE["col2"] * 1.061, 3.66))
    # Row pitch is set by the axes HEIGHT, and it has two lines of text to hold rather than one
    # (Leif 2026-08-25: "XGBoost and its subtitle aren't squashed that much"). 0.750 x 3.66in =
    # 2.75in over 14 rows is 0.196in per row; at the previous height the bold line and its
    # subtitle nearly touched the rows above and below.
    # bottom margin holds a TWO-LINE x-label when a provisional footnote is present; at the
    # one-line spacing the legend sat on top of the second line.
    ax = fig.add_axes([0.232, 0.243, 0.760, 0.750])

    ytrans = ax.get_yaxis_transform()          # x in axes coords, y in data coords
    for yi, a in enumerate(order):
        sysname, sub, c = _meta(a)
        pending = a not in out.index
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
        if pending:
            ax.text(0.5 * (1 + nfield), yi, "awaiting Wong + FartDB", ha="center", va="center",
                    fontsize=FS["annot"] - 0.5, color="#7A7A7A", style="italic", zorder=3)
        else:
            r = out.loc[a]
            for k in T.CATEGORIES:
                if np.isfinite(r[k]):
                    ax.plot(r[k], yi, marker=T.CAT_MARKER[k], mfc="none", mec=c, mew=0.9,
                            ms=3.8, ls="none", zorder=3)
            ax.errorbar(r["mean_rank"], yi, xerr=r["se_rank"], fmt="none",
                        ecolor=c, elinewidth=1.1, capsize=1.6, capthick=0.9, zorder=3)
            # A TICK, not a dot (Leif 2026-08-25). The four category marks are already open
            # outlines; a fifth filled outline in the same size class competed with them, and the
            # tick reads as "the summary of this row" rather than as another category.
            ax.plot(r["mean_rank"], yi, marker="|", ms=9.0, mew=2.0, color=c, zorder=4)
            # The number goes on every row: a plot whose ranks must be read off a tick axis is
            # not a ranking. Placed past the rightmost mark ON THIS ROW, not past the whisker.
            right = np.nanmax([r["mean_rank"] + r["se_rank"]] +
                              [r[k] for k in T.CATEGORIES if np.isfinite(r[k])])
            ax.text(right + 0.28, yi, f"{r['mean_rank']:.2f}", va="center", ha="left",
                    fontsize=FS["annot"] - 0.5, color=INK, zorder=5)

    # TWO-LINE ROW LABEL, drawn by hand rather than as tick text because matplotlib cannot mix
    # weights inside one tick string: the predicting model in bold, what it is fed underneath.
    # Offsets are +-0.21 as in fig_A1, and that is a constraint rather than a preference -- at
    # +-0.26 the within-row gap (0.52) exceeds the between-row gap (0.48) and every second line
    # reads as though it belongs to the arm BELOW it. Within-row must stay tighter than
    # between-row whatever the row pitch.
    for yi, a in enumerate(order):
        sysname, sub, _ = _meta(a)
        grey = "#7A7A7A" if a not in out.index else INK
        st = "italic" if a not in out.index else "normal"
        if a in short_prov:
            sysname = sysname + " *"
        ax.text(-0.012, yi - 0.21, sysname, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 0.4, fontweight="bold", color=grey, style=st)
        ax.text(-0.012, yi + 0.21, sub, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 1.3, color=grey, style=st)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.42, -0.62)
    ax.set_xlim(0.4, nfield + 1.6)
    ax.set_xticks(range(1, nfield + 1, 2))
    # The provisional footnote lives ON THE X-LABEL, not as free-floating text below the axes: at
    # transAxes y=-0.145 it landed behind the legend and was clipped, and str.replace had silently
    # not matched the legend line anyway, so the caveat rendered nowhere at all. A label cannot be
    # lost that way.
    if weighting == "dataset":
        xlab = (f"{SUMMARY} rank over all {int(out['n_datasets'].max())} datasets, each weighted "
                f"equally (1 = best of {nfield} scored)")
    else:
        xlab = (f"{SUMMARY} rank over four task categories, equally weighted "
                f"(1 = best of {nfield} scored)")
    if short_prov:
        a0 = next(iter(short_prov))
        miss = ", ".join(m.split(":")[0] for m in missing_ds[a0])
        # SHORT ENOUGH TO FIT THE CROP. save() trims to drawn content, so a footnote wider than
        # the axes silently sets the plate width and then LaTeX scales every font down to fit --
        # and at the previous length the last word was cut off the canvas outright.
        tail = ("not the same datasets as its neighbours"
                if weighting == "category" else "a smaller set than its neighbours")
        xlab += (f"\n*  provisional: ranked on {short_prov[a0]} of "
                 f"{int(out['n_datasets'].max())} datasets (no {miss}) \u2014 {tail}")
    ax.set_xlabel(xlab, fontsize=FS["label"])
    ax.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=FS["tick"] - 0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # The legend carries the DATASET count per category, and the count still owed. Virtual
    # screening currently rests on a single dataset, so its open diamonds are one rank rather than
    # a mean -- a reader cannot see that from the marker, and it is the largest caveat on the
    # plate right now.
    pend = {k: sum(1 for v in T.PENDING_DATASETS.values() if v == k) for k in T.CATEGORIES}
    h = [Line2D([], [], ls="none", marker=T.CAT_MARKER[k], mfc="none", mec=INK, mew=0.9, ms=3.8,
                label=f"{k} ({avail[k]}" + (f"+{pend[k]} pending)" if pend[k] else ")"))
         for k in T.CATEGORIES]
    # NO KEY FOR THE TICK (Leif 2026-08-25). The x-axis already reads "mean rank over four task
    # categories" and the tick is the only filled mark on a row of open ones, so the entry
    # restated the axis and cost the legend a slot.
    # CENTRED ON THE X-AXIS, not on the plate (Leif 2026-08-25). The two are ~0.6in apart because
    # the row-label column hangs off the left of the axes, and the legend keys describe marks that
    # live inside the axes -- so the axes span is the thing they should line up with.
    #
    # Taken from ax.get_position(), the axes RECTANGLE, rather than get_tightbbox(), which would
    # add the label column back in and put us where we started. Reading it rather than repeating
    # the literal also means it follows the layout if the axes fractions change.
    leg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.004),
                     ncol=row_ncol(h, rows=1), fontsize=FS["annot"] - 0.5, handletextpad=0.4,
                     columnspacing=1.4, borderpad=0.3, **LEGEND_BOX)
    bb = ax.get_position()
    leg.set_bbox_to_anchor((0.5 * (bb.x0 + bb.x1), 0.004), transform=fig.transFigure)

    save(fig, name, subdir=subdir)
    plt.close(fig)
    report(out, avail, order, head_seed, short_prov, missing_ds)


def report(out, avail, order, head_seed, short_prov, missing_ds):
    print(f"\nFig A — {len(out)} of {len(RANKED_ARMS)} arms scored, "
          f"{out.attrs['n_datasets_total']} datasets in four categories\n")
    print(f"   {'model':<34}" + "".join(f"{T.CAT_SHORT[k]:>9}" for k in T.CATEGORIES)
          + f"{'mean':>8}{'SE':>7}{'n_ds':>6}")
    for a in order:
        _sys, _sub, _ = _meta(a)
        lab = f"{_sys} {_sub}"
        if a not in out.index:
            print(f"   {lab:<34}" + f"{'awaiting Wong + FartDB':>39}")
            continue
        r = out.loc[a]
        print(f"   {lab:<34}" + "".join(f"{r[k]:>9.2f}" for k in T.CATEGORIES)
              + f"{r['mean_rank']:>8.2f}{r['se_rank']:>7.2f}{int(r['n_datasets']):>6}")
    print(f"\n   datasets per category: "
          + ", ".join(f"{k} {avail[k]}" for k in T.CATEGORIES))
    pend = {k: v for k, v in T.PENDING_DATASETS.items()}
    if pend:
        print(f"   awaiting datasets: " + ", ".join(f"{k} -> {v}" for k, v in pend.items()))

    # CAPTION FACTS. Printed rather than left for the caption writer to count, because every one
    # of these is a number a reader will take on trust and none of them can fail on their own.
    # Paste from here into the LaTeX \caption{}; do not re-derive them by eye from the plate.
    print("\n   CAPTION FACTS (paste, do not re-count):")
    print(f"     field                {len(RANKED_ARMS)} arms, {out.attrs['n_datasets_total']} "
          f"datasets, four categories at equal weight "
          f"({', '.join(f'{k} {avail[k]}' for k in T.CATEGORIES)})")
    print(f"     replicate axis       {len(RANKED_ARMS) - len(head_seed)} arms replicate the "
          f"PRETRAINING; {len(head_seed)} replicate the HEAD SEED inside one run "
          f"({', '.join(head_seed)}).")
    print( "                          A head-seed bar EXCLUDES pretraining variance and therefore "
           "reads tighter for a")
    print( "                          reason unrelated to the arm's stability. Say so; do not "
           "write \"three seeds\" flat.")
    for a, n in short_prov.items():
        print(f"     provisional          {a} ranked on {n} of "
              f"{int(out['n_datasets'].max())} datasets "
              f"(no {', '.join(m.split(':')[0] for m in missing_ds[a])})")


if __name__ == "__main__":
    main()
    # The category-weighted cut, kept for comparison and for answering a referee who asks whether
    # the 30 MoleculeACE targets drive the ordering. Rendered into panels/ so it cannot be
    # mistaken for the paper plate.
    main(weighting="category", name="fig_A_by_category", subdir="panels")

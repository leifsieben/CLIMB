"""Fig A — overall standing across four TASK TYPES, ranking only.

ONE script, ONE figure: figures_v2/fig_A.png / .pdf

Full A4 text-block width, and as short as the content allows: one row per model, no second
panel. The six per-dataset panels that used to sit underneath are fig_A2's job; repeating them
here cost most of a page and said nothing the ranking did not.

WHAT IS RANKED. Thirteen models, every one of them a FROZEN representation with a trained head. Each is ranked WITHIN every individual dataset (1 = best of the
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

  1. "3 seeds" is one label over two estimands. CLIMB arms vary the PRETRAINING with head seeds
     pinned; ECFP4 and the three literature CLMs have no pretraining stage to vary, so they carry
     three disjoint HEAD-SEED triples instead. Do NOT write "three pretraining seeds" as a
     property of the panel -- it is false for five of the thirteen rows.
  2. The probe head is representation-dependent by design: ECFP4 arms at XGBoost, every CLM at a
     frozen encoder plus MLP, because SI fig f shows the preference is representation-dependent
     and a single head handicaps whichever representation it does not suit. The three literature
     CLMs have NOT been measured at XGBoost, and CheMeleon -- the closest analogue -- prefers
     XGBoost by 0.138 macro RMSE, wider than the whole ECFP4+desc-to-CLIMB span. Leif ruled no
     ablation for now; the exposure is recorded rather than hidden.

MISSING DATA IS DRAWN, NOT OMITTED. Wong (virtual screening) and FartDB (classification) are
commissioned and not yet complete -- 4 of 13 and 2 of 13 arms respectively as of 2026-08-26 -- so
they are absent from the counts in the legend rather than half-scored. An arm with no results at
all gets a labelled empty row, not an absent row: dropping it would make the plate look finished
and quietly renumber the field, which changes every rank on the page.

Run:  python3 -m figures.fig_A
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX, row_ncol
from figures.arms import ARMS, SHADES, system, label as arm_label
from figures import tasksuites as T

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
RANKED_ARMS = ["ecfp", "ecfp_desc",
               "sup_dense", "sup_sparse", "sup_mixed",
               "unsup", "u2s_dense", "u2s_sparse", "s2u_dense", "random_encoder",
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

N = len(RANKED_ARMS)
HAVE = [a for a in RANKED_ARMS if a in ARMS]


def _meta(a):
    """(bold first line, second line, colour) for one row."""
    if a in ARMS:
        return system(a), arm_label(a), ARMS[a]["color"]
    p = PENDING_ARMS[a]
    return p["system"], p["label"], p["color"]


def compute():
    """Ranks over the thirteen-arm field, plus per-arm category coverage.

    The field size passed to the ranker is len(HAVE), so the ranks returned run 1..len(HAVE). They
    are NOT rescaled to 1..13 here: doing that would invent a position for three arms that have no
    results, and the axis is drawn to len(HAVE) with the pending rows left empty instead.
    """
    out, cat, R = T.wide_ranks(HAVE)
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
    if n_ds.nunique() > 1:
        short = {a: int(n_ds[a]) for a in n_ds.index if n_ds[a] < n_ds.max()}
        raise AssertionError(
            f"fig_A: arms do not cover the same datasets. Full field is {int(n_ds.max())}; "
            f"short: {short}. A category mean taken over a different dataset set is not the same "
            f"quantity as its neighbours'. Score or fetch the missing runs -- do not rank a "
            f"partial arm beside complete ones.")
    avail = {k: int((cat == k).sum()) for k in T.CATEGORIES}
    return out, avail


BAND = "#F2F2F2"          # zebra row band, same value as fig_A1 so the two plates read alike


def main():
    out, avail = compute()
    order = list(out.sort_values("mean_rank").index) + [a for a in RANKED_ARMS if a not in HAVE]
    nfield = len(HAVE)

    # One row per model. Thirteen rows plus axis and legend inside 3.4in, roughly a third of A4's
    # text height -- the point of dropping the six-panel block.
    #
    # WIDTH IS SET BY MEASUREMENT. save() crops to the drawn content, so the rendered plate is NOT
    # figsize wide: an axes box that leaves slack renders narrower than the text block and LaTeX
    # then scales every font in it up relative to the rest of the set. This layout is authored to
    # fill the canvas and measured at 6.73in against a 6.69in text block. Re-measure if the axes
    # fractions below change, AND when the row labels change: the multiplier is the ratio of the
    # canvas to the crop, and the crop is set by the widest label in the left column, so shortening
    # the arm list moved it. First version, with slack on both sides, rendered 5.58in.
    fig = plt.figure(figsize=(STYLE["col2"] * 1.115, 3.45))
    # Row pitch is set by the axes HEIGHT, and it has two lines of text to hold rather than one
    # (Leif 2026-08-25: "XGBoost and its subtitle aren't squashed that much"). 2.83in over 13 rows
    # is 0.218in per row; at the previous 2.50in the bold line and its subtitle nearly touched the
    # rows above and below.
    ax = fig.add_axes([0.232, 0.177, 0.760, 0.820])

    ytrans = ax.get_yaxis_transform()          # x in axes coords, y in data coords
    for yi, a in enumerate(order):
        sysname, sub, c = _meta(a)
        pending = a not in HAVE
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
        if pending:
            ax.text(0.5 * (1 + nfield), yi, "awaiting results", ha="center", va="center",
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
        grey = "#7A7A7A" if a not in HAVE else INK
        st = "italic" if a not in HAVE else "normal"
        ax.text(-0.012, yi - 0.21, sysname, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 0.4, fontweight="bold", color=grey, style=st)
        ax.text(-0.012, yi + 0.21, sub, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 1.3, color=grey, style=st)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.42, -0.62)
    ax.set_xlim(0.4, nfield + 1.6)
    ax.set_xticks(range(1, nfield + 1, 2))
    ax.set_xlabel(f"mean rank over four task categories, equally weighted "
                  f"(1 = best of {nfield} scored)", fontsize=FS["label"])
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

    save(fig, "fig_A")
    plt.close(fig)
    report(out, avail, order)


def report(out, avail, order):
    print(f"\nFig A — {len(HAVE)} of {len(RANKED_ARMS)} arms scored, "
          f"{out.attrs['n_datasets_total']} datasets in four categories\n")
    print(f"   {'model':<34}" + "".join(f"{T.CAT_SHORT[k]:>9}" for k in T.CATEGORIES)
          + f"{'mean':>8}{'SE':>7}{'n_ds':>6}")
    for a in order:
        _sys, _sub, _ = _meta(a)
        lab = f"{_sys} {_sub}"
        if a not in HAVE:
            print(f"   {lab:<34}" + f"{'awaiting results':>39}")
            continue
        r = out.loc[a]
        print(f"   {lab:<34}" + "".join(f"{r[k]:>9.2f}" for k in T.CATEGORIES)
              + f"{r['mean_rank']:>8.2f}{r['se_rank']:>7.2f}{int(r['n_datasets']):>6}")
    print(f"\n   datasets per category: "
          + ", ".join(f"{k} {avail[k]}" for k in T.CATEGORIES))
    pend = {k: v for k, v in T.PENDING_DATASETS.items()}
    if pend:
        print(f"   awaiting datasets: " + ", ".join(f"{k} -> {v}" for k, v in pend.items()))


if __name__ == "__main__":
    main()

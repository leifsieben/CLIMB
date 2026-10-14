from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX, row_ncol
from figures.arms import ARMS, system, label as arm_label, assert_not_retired
from figures import tasksuites as T
from figures.allsuites import wide_table as A_wide

check_font()
INK = "#000000"


RANKED_ARMS = ["ecfp", "ecfp_desc",
               "sup_dense", "sup_sparse", "sup_dense_sparse",
               "unsup", "u2s_dense", "u2s_sparse", "s2u_dense", "random_encoder",


               "unsup_100M",


               "sup_dense_100M",
               "chemberta_mtr", "molformer_c3", "selfies_ted"]

assert_not_retired(RANKED_ARMS, "figure_1.RANKED_ARMS")


PENDING_ARMS = {}
assert not (set(PENDING_ARMS) & set(ARMS)), \
    "an arm is declared PENDING here and also present in arms.py -- promote it and delete it here"
assert set(RANKED_ARMS) - set(ARMS) == set(PENDING_ARMS), \
    f"RANKED_ARMS names arms that are neither in arms.py nor declared pending: " \
    f"{sorted(set(RANKED_ARMS) - set(ARMS) - set(PENDING_ARMS))}"


PROVISIONAL = {"unsup_100M"}


AWAITING_DATA = set()

N = len(RANKED_ARMS)
HAVE = [a for a in RANKED_ARMS if a in ARMS and a not in AWAITING_DATA]


STARRED = {a for a in RANKED_ARMS
           if a in ARMS and ARMS[a].get("pretrain_replicates") is False
           and ARMS[a].get("system", "").startswith("CLIMB 100M")}


def _meta(a):
    """(bold first line, second line, colour) for one row."""
    if a in ARMS:


        name = system(a) + ("*" if a in STARRED else "")
        return name, arm_label(a), ARMS[a]["color"]
    p = PENDING_ARMS[a]
    return p["system"], p["label"], p["color"]


def compute():


    full_S, _full_M = A_wide(RANKED_ARMS)
    n_by_arm = full_S.notna().sum(axis=1)
    target = int(n_by_arm.max())
    really_short = {a for a in RANKED_ARMS if a in ARMS and int(n_by_arm.get(a, 0)) < target}
    stale = AWAITING_DATA - really_short
    undeclared = really_short - AWAITING_DATA - PROVISIONAL
    assert not stale, (
        f"figure_1: {sorted(stale)} is declared AWAITING_DATA but now covers all {target} datasets. "
        f"Remove it from AWAITING_DATA so it is ranked -- an empty row drawn over real results is "
        f"worse than no row.")
    assert not undeclared, (
        f"figure_1: {sorted(undeclared)} is short of the {target}-dataset field and is neither "
        f"AWAITING_DATA nor PROVISIONAL. Declare it or score it; do not rank it.")

    out, cat, R = T.wide_ranks(HAVE, summary=SUMMARY)
    missing = [a for a in HAVE if a not in out.index]
    assert not missing, f"figure_1: arms in RANKED_ARMS with no row in the score table: {missing}"


    n_ds = out["n_datasets"].astype(int)
    full = int(n_ds.max())
    short_prov = {a: int(n_ds[a]) for a in n_ds.index if a in PROVISIONAL and n_ds[a] < full}
    missing_ds = {}
    for a in short_prov:
        have = set(R.columns[R.loc[a].notna()])
        missing_ds[a] = sorted(set(R.columns) - have)
        print(f"   [figure_1] {a} is PROVISIONAL, ranked on {short_prov[a]} of {full} datasets, "
              f"missing {missing_ds[a]}. Marked * on the plate.")
    n_ds = n_ds.drop(index=list(short_prov))
    if n_ds.nunique() > 1:
        short = {a: int(n_ds[a]) for a in n_ds.index if n_ds[a] < n_ds.max()}
        raise AssertionError(
            f"figure_1: arms do not cover the same datasets. Full field is {int(n_ds.max())}; "
            f"short: {short}. A category mean taken over a different dataset set is not the same "
            f"quantity as its neighbours'. Score or fetch the missing runs -- do not rank a "
            f"partial arm beside complete ones.")
    avail = {k: int((cat == k).sum()) for k in T.CATEGORIES}
    return out, avail, short_prov, missing_ds


BAND = "#F2F2F2"


SUMMARY = "mean"


def main():
    out, avail, short_prov, missing_ds = compute()
    out = out.assign(mean_rank=out["mean_rank_pooled"], se_rank=out["se_rank_pooled"])
    order = list(out.sort_values("mean_rank").index) + [a for a in RANKED_ARMS if a not in out.index]
    nfield = len(out)


    fig = plt.figure(figsize=(STYLE["col2"] * 1.061, 3.66))


    ax = fig.add_axes([0.232, 0.243, 0.760, 0.750])

    ytrans = ax.get_yaxis_transform()
    for yi, a in enumerate(order):
        sysname, sub, c = _meta(a)
        pending = a not in out.index
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
        if pending:
            ax.text(0.5 * (1 + nfield), yi, "awaiting Wong + FartDB", ha="center", va="center",
                    fontsize=FS["annot"] - 0.5, color=INK, style="italic", zorder=3)
        else:
            r = out.loc[a]
            for k in T.CATEGORIES:
                if np.isfinite(r[k]):
                    ax.plot(r[k], yi, marker=T.CAT_MARKER[k], mfc="none", mec=c, mew=0.9,
                            ms=3.8, ls="none", zorder=3)
            ax.errorbar(r["mean_rank"], yi, xerr=r["se_rank"], fmt="none",
                        ecolor=c, elinewidth=1.1, capsize=1.6, capthick=0.9, zorder=3)


            ax.plot(r["mean_rank"], yi, marker="|", ms=9.0, mew=2.0, color=c, zorder=4)


            right = np.nanmax([r["mean_rank"] + r["se_rank"]] +
                              [r[k] for k in T.CATEGORIES if np.isfinite(r[k])])
            ax.text(right + 0.28, yi, f"{r['mean_rank']:.2f}", va="center", ha="left",
                    fontsize=FS["annot"] - 0.5, color=INK, zorder=5)


    for yi, a in enumerate(order):
        sysname, sub, _ = _meta(a)


        st = "italic" if a not in out.index else "normal"
        if a in short_prov:
            sysname = sysname + " *"
        ax.text(-0.012, yi - 0.21, sysname, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 0.4, fontweight="bold", color=INK, style=st)
        ax.text(-0.012, yi + 0.21, sub, transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"] - 1.3, color=INK, style=st)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.42, -0.62)
    ax.set_xlim(0.4, nfield + 1.6)
    ax.set_xticks(range(1, nfield + 1, 2))


    xlab = (f"{SUMMARY} rank over all {int(out['n_datasets'].max())} datasets, each weighted "
            f"equally (1 = best of {nfield} scored)")
    if short_prov:
        a0 = next(iter(short_prov))
        miss = ", ".join(m.split(":")[0] for m in missing_ds[a0])


        xlab += (f"\n*  provisional: ranked on {short_prov[a0]} of "
                 f"{int(out['n_datasets'].max())} datasets (no {miss}) \u2014 "
                 "a smaller set than its neighbours")


    ax.set_xlabel(xlab, fontsize=FS["label"])
    ax.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=FS["tick"] - 0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


    pend = {k: sum(1 for v in T.PENDING_DATASETS.values() if v == k) for k in T.CATEGORIES}
    h = [Line2D([], [], ls="none", marker=T.CAT_MARKER[k], mfc="none", mec=INK, mew=0.9, ms=3.8,
                label=f"{k} ({avail[k]}" + (f"+{pend[k]} pending)" if pend[k] else ")"))
         for k in T.CATEGORIES]


    leg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.030),
                     ncol=row_ncol(h, rows=1), fontsize=FS["annot"] - 0.5, handletextpad=0.4,
                     columnspacing=1.4, borderpad=0.3, **LEGEND_BOX)
    bb = ax.get_position()
    leg.set_bbox_to_anchor((0.5 * (bb.x0 + bb.x1), 0.030), transform=fig.transFigure)

    save(fig, "figure_1")
    plt.close(fig)


if __name__ == "__main__":
    main()

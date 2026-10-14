from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX
from figures.arms import ARMS, system, label as arm_label, assert_not_retired
from figures import allsuites as A
from figures import tasksuites as T

check_font()
INK = "#000000"


ARM_ORDER = ["ecfp", "ecfp_desc",
             "unsup_100M", "sup_dense_100M",
             "chemberta_mtr", "molformer_c3", "selfies_ted"]

assert_not_retired(ARM_ORDER, "figure_C1.ARM_ORDER")


PANELS = [
    ("Activity cliffs",   "rmse",                "macro RMSE",            False),
    ("Regression",        "spearmanr",           "Spearman ρ",            True),
    ("Regression",        "mean_absolute_error", "MAE",                   False),
    ("Classification",    "roc_auc",             "ROC-AUC",               True),
    ("Classification",    "pr_auc",              "PR-AUC",                True),
    ("Virtual screening", "nef1",                "NEF1%",                 True),
]


METRIC_ALIASES = {("Classification", "roc_auc"): {"macro_ovr_auc"}}


PANEL_H = 0.80
TITLE_IN = 0.22
TOP_IN = 0.05
NOTE_IN = 0.145
GAP_IN = 0.05
PAD_IN = 0.02
WSPACE = 0.62
LEFT_IN = 0.44

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


        S, M = A.wide_table(have, metrics={c: metric for c in in_cat})
        ok = set(METRIC_ALIASES.get((name, metric), set())) | {metric}
        cols = [c for c in in_cat if c in S.columns and M.loc[c, "metric"] in ok]
        assert cols, f"figure_C1: no {name} dataset could be read on {metric!r}"

        offscale = []
        if metric in SCALE_GUARDED and len(cols) > 2:
            field = S[cols].mean(axis=0)
            med = float(field.median())
            offscale = sorted(c for c in cols
                              if med > 0 and float(field[c]) / med > SCALE_RATIO)
            cols = [c for c in cols if c not in offscale]

        sub = S.loc[[a for a in ARM_ORDER if a in S.index], cols]
        n = sub.notna().sum(axis=1)
        d = pd.DataFrame({"mean": sub.mean(axis=1),
                          "se": sub.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1)),
                          "n": n})
        d.loc[n < 2, "se"] = np.nan
        d["n_cat"] = len(in_cat)
        d["n_used"] = len(cols)
        d.attrs["dropped"] = sorted(set(in_cat) - set(cols) - set(offscale))
        d.attrs["offscale"] = offscale
        out[(name, metric)] = d
    return out


def _notes(res):

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


ABBREV = {"unsupervised": "unsup.", "supervised": "sup.",
          "supervised, desc": "sup. desc", "supervised, desc+sparse": "sup. desc+sparse"}


def _legend_label(a):
    """Compact key text, built from the arm's fields rather than written out per arm."""
    sysname, sub, _ = _meta(a)
    if a in ARMS and ARMS[a].get("objective"):
        return sysname
    return f"{sysname}, {ABBREV.get(sub, sub)}"


def _meta(a):
    return system(a), arm_label(a), ARMS[a]["color"]


def main():
    res = compute()


    notes = _notes(res)
    fig_w = STYLE["col2"] * 0.983
    fig = plt.figure(figsize=(fig_w, 3.0))
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
                        fontsize=FS["annot"] - 2.5, color=INK, style="italic", zorder=4)
                continue
            r = d.loc[a]
            ax.bar(xi, r["mean"], width=0.84, color=c, edgecolor="none", zorder=2)
            if np.isfinite(r["se"]):
                ax.errorbar(xi, r["mean"], yerr=r["se"], fmt="none", ecolor=INK,
                            elinewidth=0.7, capsize=1.2, capthick=0.6, zorder=3)


            if int(r["n"]) < int(d["n_used"].iloc[0]):
                top = r["mean"] + (r["se"] if np.isfinite(r["se"]) else 0.0)
                ax.annotate("*", (xi, top), textcoords="offset points", xytext=(0, 1.0),
                            ha="center", va="bottom", fontsize=FS["annot"], color=INK, zorder=5)
                marked = True


        if marked:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.75, len(ARM_ORDER) - 0.25)
        arrow = "↑" if higher else "↓"
        n_used, n_cat = int(d["n_used"].iloc[0]), int(d["n_cat"].iloc[0])


        head = f"{name} {arrow}"
        prev_cat = name
        ax.set_title(head, fontsize=FS["title"] - 1.5, fontweight="bold", color=INK, pad=3)


        ax.set_ylabel(f"{ylab}  ({n_used}/{n_cat})", fontsize=FS["annot"] - 1.5)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=FS["tick"] - 2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    handles = []
    for a in ARM_ORDER:
        sysname, sub, c = _meta(a)
        handles.append(Patch(facecolor=c, edgecolor="none", alpha=1.0,
                             label=_legend_label(a)))


    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=len(handles),
                     fontsize=FS["legend"] - 1.8, handlelength=0.95, handletextpad=0.4,
                     columnspacing=0.9, labelspacing=0.28, borderpad=0.28, **LEGEND_BOX)

    fig.canvas.draw()
    leg_w = leg.get_window_extent().width / fig.dpi
    if leg_w > STYLE["col2"] + 0.02:
        print(f"  [figure_C1] WARNING legend is {leg_w:.2f}in against a {STYLE['col2']}in text "
              f"block -- it, not the axes, will set the plate width and LaTeX will scale every "
              f"font down. Shorten the keys or drop to two rows.")
    leg_in = leg.get_window_extent().height / fig.dpi


    fig_h = TOP_IN + TITLE_IN + PANEL_H + GAP_IN + leg_in + PAD_IN
    fig.set_size_inches(fig_w, fig_h)


    leg.set_bbox_to_anchor((0.5, PAD_IN / fig_h), transform=fig.transFigure)


    fig.canvas.draw()
    leg_top_in = leg.get_window_extent().y1 / fig.dpi
    fig.subplots_adjust(
        left=LEFT_IN / fig_w, right=1 - 0.02 / fig_w,
        bottom=(leg_top_in + GAP_IN) / fig_h,
        top=1 - (TOP_IN + TITLE_IN) / fig_h,
        wspace=WSPACE)
    save(fig, "figure_C1")
    plt.close(fig)
    report(res)
    if notes:
        print("\n  FOR THE CAPTION -- derived from the table, not drawn on the plate:")
        for line in notes:
            print(f"    {line}")


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

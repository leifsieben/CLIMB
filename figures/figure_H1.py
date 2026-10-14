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


LINES = ["e2e_no_pretrain", "sup_dense", "unsup", "ecfp_desc"]
MARKER = {"e2e_no_pretrain": "o", "sup_dense": "s", "unsup": "D", "ecfp_desc": "^"}
_unmarked = [a for a in LINES if a not in MARKER]
assert not _unmarked, f"LINES has {_unmarked} with no marker -- add one to MARKER"

YMARGIN = 0.18


def _fmt_n(v, _):
    return f"{v/1000:.1f}k" if v >= 1000 else f"{v:g}"


def main():


    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.7))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"

        title = g_all.task.iloc[0] if not g_all.empty else d["label"]
        ax.set_title(f"{title} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)


        pass
        ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if g_all.empty:
            ax.text(0.5, 0.5, "no label-fraction\nsweep run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue


        if p == "MoleculeACE":
            ax.text(0.98, 0.96, "band = spread across\nthe 30 targets", transform=ax.transAxes,
                    ha="right", va="top", fontsize=FS["annot"] - 1.5, color=STYLE["mute"],
                    style="italic", linespacing=1.15)

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


            e = g.sd.to_numpy(dtype=float)
            v = g.value.to_numpy(dtype=float)


            ok = np.isfinite(e)
            if ok.any():
                ax.fill_between(g.n_train.to_numpy()[ok], (v - e)[ok], (v + e)[ok],
                                color=ARMS[arm]["color"], alpha=0.16, lw=0, zorder=1)

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


    fig.tight_layout(rect=(0, 0.115, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.096, "labelled training molecules", ha="center", va="bottom",
             fontsize=FS["annot"], color=INK)
    save(fig, "figure_H1")
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

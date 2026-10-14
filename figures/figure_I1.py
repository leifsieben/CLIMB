from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX
from figures.arms import PANELS, PANEL_ORDER, SHADES
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_d" / "SI_fig_d_augmentation.csv")


MODES = [("canonical", SHADES["sup"][0], "o"), ("augmented", SHADES["sup"][2], "D")]
YMARGIN = 0.22


def main():


    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.65))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)


        pass
        ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if g_all.empty:
            ax.text(0.5, 0.5, "not run on this\nprotocol", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lo, hi = np.inf, -np.inf
        for mode, colour, marker in MODES:
            g = g_all[g_all["mode"] == mode].sort_values("fraction")
            if g.empty:
                continue


            sd = pd.to_numeric(g.sd, errors="coerce").to_numpy()
            ax.errorbar(g.fraction, g.value, yerr=sd, color=colour, ls="-", lw=STYLE["lw"],
                        marker=marker, ms=4.6, mec="white", mew=0.6,
                        elinewidth=1.0, capsize=2.2, capthick=1.1, ecolor=INK, zorder=3)


            lo = min(lo, np.nanmin(np.where(np.isnan(sd), g.value, g.value - sd)))
            hi = max(hi, np.nanmax(np.where(np.isnan(sd), g.value, g.value + sd)))

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.001, 0.01, 0.1, 1.0]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: {0.001: "0.1%", 0.01: "1%", 0.1: "10%", 1.0: "100%"}.get(v, f"{v:g}")))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(0.0006, 1.7)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=c, marker=m, ms=4.5, lw=1.2, label=lab)
               for lab, c, m in MODES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.064),
               ncol=row_ncol(handles), fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)


    fig.tight_layout(rect=(0, 0.128, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.082, "pretraining corpus fraction", ha="center", va="bottom",
             fontsize=FS["annot"], color=INK)
    save(fig, "figure_I1")
    plt.close(fig)

    print("\nSI Fig d — augmented minus canonical (+ = augmented better):")
    for p in PANEL_ORDER:
        g_all = DF[DF.panel == p]
        if g_all.empty:
            print(f"   {p:<12} — not run on this protocol")
            continue
        sign = 1 if g_all.higher_better.iloc[0] else -1
        cells = []
        for frac in sorted(g_all.fraction.unique()):
            c = g_all[(g_all["mode"] == "canonical") & (g_all.fraction == frac)]
            a = g_all[(g_all["mode"] == "augmented") & (g_all.fraction == frac)]
            if not len(c) or not len(a):
                continue
            delta = sign * (float(a.value.iloc[0]) - float(c.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(c.sd, errors="coerce").iloc[0],
                          pd.to_numeric(a.sd, errors="coerce").iloc[0])
            cells.append(f"{frac:>6g}:{delta:+8.4f}{'*' if np.isfinite(sd) and abs(delta) > sd else ' '}")
        print(f"   {p:<12} " + " ".join(cells))
    print("   * = |delta| exceeds the combined pretraining-seed SD")


if __name__ == "__main__":
    main()

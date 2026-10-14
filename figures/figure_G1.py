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

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_b" / "SI_fig_b_vocab.csv")


FAMILIES = [("BPE", SHADES["unsup"][0], "o"), ("Unigram", SHADES["unsup"][2], "D")]
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
            ax.text(0.5, 0.5, "not run in the\nvocabulary wave", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lo, hi = np.inf, -np.inf
        for fam, colour, marker in FAMILIES:
            g = g_all[g_all.family == fam].sort_values("vocab")
            if g.empty:
                continue


            sd = pd.to_numeric(g.sd, errors="coerce").to_numpy()
            ax.errorbar(g.vocab, g.value, yerr=sd, color=colour, ls="-", lw=STYLE["lw"],
                        marker=marker, ms=4.6, mec="white", mew=0.6,
                        elinewidth=1.0, capsize=2.2, capthick=1.1, ecolor=INK, zorder=3)


            lo = min(lo, np.nanmin(np.where(np.isnan(sd), g.value, g.value - sd)))
            hi = max(hi, np.nanmax(np.where(np.isnan(sd), g.value, g.value + sd)))

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator([261, 1000, 3000, 12000]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlim(200, 16000)
        pad = YMARGIN * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=c, marker=m, ms=4.5, lw=1.2, label=f)
               for f, c, m in FAMILIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.064),
               ncol=row_ncol(handles), fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)


    fig.tight_layout(rect=(0, 0.128, 1, 1), w_pad=0.35)
    fig.text(0.5, 0.082, "tokenizer vocabulary", ha="center", va="bottom",
             fontsize=FS["annot"], color=INK)
    save(fig, "figure_G1")
    plt.close(fig)

    print("\nSI Fig b — vocabulary spread vs replicate noise (the near-null test):")
    print(f"   {'panel':<12}{'family':<9}{'range over vocab':>18}{'median sd':>12}  verdict")
    for p in PANEL_ORDER:
        g_all = DF[DF.panel == p]
        if g_all.empty:
            print(f"   {p:<12}{'—':<9}{'not run in the vocabulary wave':>32}")
            continue
        for fam, _, _ in FAMILIES:
            g = g_all[g_all.family == fam]
            if len(g) < 2:
                continue
            rng = g.value.max() - g.value.min()
            sd = pd.to_numeric(g.sd, errors="coerce").median()
            print(f"   {p:<12}{fam:<9}{rng:>18.4f}{sd:>12.4f}  "
                  f"{'within noise' if rng <= sd else f'{rng/sd:.1f}x noise'}")


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation

from figures.style import STYLE, FS, save, check_font, LEGEND_BOX
from figures.arms import SHADES, ARMS

check_font()

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "figure_data" / "fig_E" / "fig_E_lift.csv"
TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]


PANELS = [
    ("supervised", "a", "Supervised: permuted targets",
     [("real",             "supervised, desc",                    SHADES["sup"][0]),
      ("targets_permuted", "corrupted targets",                    SHADES["sup"][2])]),
    ("unsupervised", "b", "Unsupervised: degraded corpus statistics",
     [("real",     "unsupervised",       SHADES["unsup"][0]),
      ("shuffled", "shuffled tokens",    SHADES["unsup"][1]),
      ("bigram",   "bigram corpus",      SHADES["unsup"][2]),
      ("unigram",  "unigram corpus",     SHADES["unsup"][3]),
      ("wiki",     "Wikipedia",          SHADES["random"][1])]),
]


def _lim(sub, pad_lo=0.08, pad_hi=0.22):

    lo = min((sub.lift_pct - sub.lift_sd_pct.fillna(0)).min(), 0)
    hi = (sub.lift_pct + sub.lift_sd_pct.fillna(0)).max()
    sp = hi - lo
    return lo - pad_lo * sp, hi + pad_hi * sp


def draw(fig, ax, d, series, tag, subtitle, ylim, compact=False):

    x = np.arange(len(TASKS))
    n = len(series)
    w = 0.80 / n
    for i, (key, label, colour) in enumerate(series):
        s = d[d.arm == key].set_index("dataset")
        ys = [s.lift_pct.get(t, np.nan) for t in TASKS]


        es = [s.lift_sd_pct.get(t, np.nan) for t in TASKS]
        off = (i - (n - 1) / 2) * w

        ax.bar(x + off, ys, width=w, color=colour, edgecolor=STYLE["ink"], linewidth=0.8,
               yerr=es, error_kw=dict(elinewidth=1.0, capsize=2.2, capthick=1.1,
                                      ecolor=STYLE["ink"], zorder=6),
               label=label, zorder=3)

    ax.axhline(0, color=STYLE["ink"], lw=0.8, zorder=2)
    ax.set_xticks(x)


    ax.set_xticklabels(TASKS, rotation=22, ha="right", rotation_mode="anchor")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)


    ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontsize=FS["panel_tag"],
            fontweight="bold", va="bottom", ha="left", color=STYLE["ink"])
    ax.text(0.0, 1.03, subtitle, fontsize=FS["title"], fontweight="bold", va="bottom", ha="left",
            color=STYLE["ink"],
            transform=ax.transAxes + ScaledTranslation(13 / 72, 0, fig.dpi_scale_trans))
    if compact:


        ax.legend(loc="upper left", ncol=2, fontsize=FS["legend"] - 1,
                  handletextpad=0.4, columnspacing=0.8, borderpad=0.30, labelspacing=0.2,
                  handlelength=1.2, **LEGEND_BOX)
    else:


        ax.legend(loc="upper right", fontsize=FS["legend"], ncol=(1 if len(series) <= 2 else 2),
                  handletextpad=0.4, columnspacing=0.9, borderpad=0.35, labelspacing=0.22,
                  handlelength=1.3, **LEGEND_BOX)


def main():
    d = pd.read_csv(TABLE)


    ylims = {panel: _lim(d[d.panel == panel]) for panel, _, _, _ in PANELS}


    fig, axes = plt.subplots(1, 2, figsize=(STYLE["col2"], 2.76),
                             gridspec_kw=dict(width_ratios=[1.0, 2.05], wspace=0.24))
    for ax, (panel, tag, subtitle, series) in zip(axes, PANELS):
        draw(fig, ax, d[d.panel == panel], series, tag, subtitle, ylims[panel])
    axes[0].set_ylabel("Lift over " + ARMS["random_encoder"]["label"])
    axes[1].set_ylabel("Lift over " + ARMS["random_encoder"]["label"])

    fig.subplots_adjust(top=0.888, bottom=0.185, left=0.082, right=0.995)


    save(fig, "figure_4")
    plt.close(fig)

    for panel, _, subtitle, series in PANELS:
        p = d[d.panel == panel]
        print(f"\nFig E ({panel}) — lift % over no pretrain, random, 5-fold CV:")
        print(f"   {'arm':<38}" + "".join(f"{t:>9}" for t in TASKS))
        for key, label, _ in series:
            s = p[p.arm == key].set_index("dataset")
            row = f"   {label:<38}"
            for t in TASKS:
                v = s.lift_pct.get(t, np.nan)
                row += f"{v:>+9.1f}" if np.isfinite(v) else f"{'—':>9}"
            print(row + f"   (n_seeds={int(s.n_seeds.max())})")


if __name__ == "__main__":
    main()

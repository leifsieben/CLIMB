"""Fig E -- corrupted pretraining objectives: does the benefit survive garbling the chemistry?

ONE script, ONE figure: figures_v2/fig_E.png / .pdf   (two panels, a + b)

Both panels hold objective family, data volume, compute, schedule, architecture and probe fixed
and remove only the CHEMICAL CONTENT of the pretraining signal. Bars are lift over
`no pretrain, random` -- a random-init encoder, frozen, same probe -- so 0 means "this pretraining
objective bought nothing".

(a) SUPERVISED (descriptor regression). Real targets vs targets permuted across the batch: the
    molecule->descriptor mapping is destroyed, the target distribution is untouched.
    Reading: the real arm helps on every one of the six panels; permuting the targets does not
    merely erase the gain, it lands BELOW the untrained floor everywhere -- most sharply on the two
    panels that need genuine structure-activity signal (HIV, MoleculeACE). A supervised
    objective with no molecule->label correspondence is actively harmful, not neutral.

(b) UNSUPERVISED (MLM), a ladder of increasingly destroyed SMILES statistics:
      shuffled tokens  -- token order permuted inside each sequence (grammar gone, token
                          distribution and mask rate preserved)
      bigram corpus    -- sequences resampled from the corpus bigram statistics (local adjacency
                          only)
      unigram corpus   -- sequences resampled from the corpus unigram marginal (no structure)
      wiki             -- English Wikipedia text: real language, ZERO chemistry
    Reading: the benefit is NOT specific to real chemistry. Shuffled tokens keep most of it and are
    the BEST rung on 3 of 6 panels; Wikipedia -- with no molecules in it at all -- is positive on
    5 of 6 and actually beats the real corpus on MoleculeACE. Only the unigram rung, which destroys
    all sequential structure, collapses to the floor on every panel (-1.2 to +1.2%). What the MLM
    buys is largely a generic sequence prior, not chemical knowledge.

Together: the supervised objective's value IS its molecule->label correspondence (destroy it and
you go negative), while the unsupervised objective's value is mostly structure-agnostic.

Data / statistics
-----------------
Everything is read from `figure_data/fig_E/fig_E_lift.csv`, built by `scripts/build_fig_E_table.py`
(see that module for the full sourcing and floor argument). In brief: the paper's canonical six
panels, frozen probe; each panel lifts over the SAME random-init frozen encoder scored in that
panel's own eval wave; error bars are ONE estimand everywhere -- +-1 SD across the 3 PRETRAINING
seeds, propagated through the lift transform with the floor held fixed.

Lift over a floor is exactly scale-invariant, so the z-scored-vs-native QM7 unit split that
constrains the ABSOLUTE panels (fig_A, fig_B) cannot reach this figure.

Cells built from fewer than 2 pretraining runs are drawn WITHOUT a whisker rather than borrowing a
fold SD, which would not be the same estimand. As of 2026-08-18 that is `corrupt_mtr_8M` on
BACE / Tox21 / QM7 (its _s1/_s2 replicates have MoleculeACE / CBS / Ames but not yet the MolNet
suite) and on Ames (_s2's Polaris run is still missing). Every other cell in the figure is 3-seed.

Run:  python3 scripts/build_fig_E_table.py && python3 -m figures.fig_E
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation

from figures.style import STYLE, FS, save, check_font
from figures.arms import SHADES, ARMS

check_font()

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "figure_data" / "fig_E" / "fig_E_lift.csv"
TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]

# (arm key, legend label, colour).  Supervised = the red family; the unsupervised ladder walks the
# blue family dark->light as chemical content is removed, and the zero-chemistry Wikipedia control
# sits outside that ladder in near-black.
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


def _lim(sub, pad_lo=0.08, pad_hi=0.30):
    """Per-panel y-range. Module scope so figures/fig_E_plus_F.py reuses the identical rule."""
    lo = min((sub.lift_pct - sub.lift_sd_pct.fillna(0)).min(), 0)
    hi = (sub.lift_pct + sub.lift_sd_pct.fillna(0)).max()
    sp = hi - lo
    return lo - pad_lo * sp, hi + pad_hi * sp


def draw(fig, ax, d, series, tag, subtitle, ylim, compact=False):
    """`compact` = the assembled fig_E+F, where these panels occupy a narrow left column: the
    subtitle must not run past the axes into the neighbouring block, and the legend must not sit
    on the bars (panel b has five series against short bars)."""
    x = np.arange(len(TASKS))
    n = len(series)
    w = 0.80 / n
    for i, (key, label, colour) in enumerate(series):
        s = d[d.arm == key].set_index("dataset")
        ys = [s.lift_pct.get(t, np.nan) for t in TASKS]
        es = [s.lift_sd_pct.get(t, np.nan) for t in TASKS]
        es = [0.0 if not np.isfinite(e) else e for e in es]
        off = (i - (n - 1) / 2) * w
        # bar styling matches fig_A2: solid black edge, same error-bar weights
        ax.bar(x + off, ys, width=w, color=colour, edgecolor=STYLE["ink"], linewidth=0.8,
               yerr=es, error_kw=dict(elinewidth=1.0, capsize=2.2, capthick=1.1,
                                      ecolor=STYLE["ink"], zorder=6),
               label=label, zorder=3)

    ax.axhline(0, color=STYLE["ink"], lw=0.8, zorder=2)
    ax.set_xticks(x)
    # rotated in BOTH panels: "MoleculeACE" does not fit horizontally in panel a, and
    # rotating only one panel would make the shared category axis look like two axes.
    ax.set_xticklabels(TASKS, rotation=22, ha="right", rotation_mode="anchor")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    # panel tag and title share one baseline: the tag sits at the axes' left edge and the title is
    # offset a fixed 13 pt to its right, so the pair aligns identically in both panels regardless
    # of how wide each panel is.
    ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontsize=FS["panel_tag"],
            fontweight="bold", va="bottom", ha="left", color=STYLE["ink"])
    ax.text(0.0, 1.03, subtitle, fontsize=FS["title"], fontweight="bold", va="bottom", ha="left",
            color=STYLE["ink"],
            transform=ax.transAxes + ScaledTranslation(13 / 72, 0, fig.dpi_scale_trans))
    if compact:
        # INSIDE the axes, two columns, with extra headroom bought by the caller's ylim -- a legend
        # below the axes collided with the assembled figure's own bottom legend.
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=FS["legend"] - 1,
                  handletextpad=0.4, columnspacing=0.8, borderpad=0.0, labelspacing=0.2,
                  handlelength=1.2)
    else:
        ax.legend(loc="upper right", frameon=False, fontsize=FS["legend"],
                  ncol=1, handletextpad=0.5, borderpad=0.2, labelspacing=0.25)


def main():
    d = pd.read_csv(TABLE)

    # PER-PANEL y-ranges (user 2026-08-17). A shared range let panel a's +35% supervised bar set
    # the scale and squashed panel b's ladder, where the interesting structure (shuffled ~ real,
    # bigram partial, unigram at the floor) lives between 0 and 30%. Each panel is now scaled to
    # its own data. The axes are both "% lift over the same floor", so a reader compares by
    # reading values, not bar heights — the tick labels carry % for exactly that reason.
    ylims = {panel: _lim(d[d.panel == panel]) for panel, _, _, _ in PANELS}

    # With the per-bar labels gone, panel b no longer needs extra width to keep them apart, so the
    # ratio is set by what panel a needs for its six task labels ("MoleculeACE"/"Tox21" collide below
    # ~2.5in of axes width).
    fig, axes = plt.subplots(1, 2, figsize=(STYLE["col2"], 3.35),
                             gridspec_kw=dict(width_ratios=[1.0, 1.45], wspace=0.26))
    for ax, (panel, tag, subtitle, series) in zip(axes, PANELS):
        draw(fig, ax, d[d.panel == panel], series, tag, subtitle, ylims[panel])
    axes[0].set_ylabel("Lift over " + ARMS["random_encoder"]["label"])
    axes[1].set_ylabel("Lift over " + ARMS["random_encoder"]["label"])

    fig.subplots_adjust(top=0.905, bottom=0.155, left=0.078, right=0.995)
    # COMPONENT of fig_E+F, so it belongs in panels/ with fig_C1/C2/D and fig_A1/A2 --
    # figures_v2/ proper should hold only what goes in the paper. It is still rendered
    # standalone for review.
    save(fig, "fig_E", subdir="panels")
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

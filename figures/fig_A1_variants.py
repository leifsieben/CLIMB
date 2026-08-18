"""Fig A1 — FIVE LAYOUT VARIANTS for the user to choose between.

Renders figures_v2/figA1_v1.png ... figA1_v5.png. All five plot IDENTICAL numbers (the same
RANKS/PER_DATASET from figures.fig_A1); only the layout differs. Once a variant is chosen, its
build function is folded back into fig_A1.py and this file is deleted.

  v1  original            the current committed layout, unchanged, for reference
  v2  compact ledger      tight row pitch, values in an aligned right-hand column instead of
                          floating above each dot, hairline row rules
  v3  zebra, one line     alternating row bands, single-line "System · recipe" labels, rank
                          value at the right edge
  v4  split panel         mean rank on the left, the four per-suite ranks in their own narrow
                          companion panel so the two readings do not overlap
  v5  lollipop            stem from the axis to the dot; classical ranking-table feel

Run:  python3 -m figures.fig_A1_variants
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, system, label
from figures.fig_A1 import RANKS, SUITES, SUITE_MARKER, N, NDS, INK, build as build_v1

check_font()
ORDER = list(RANKS.index)
Y = np.arange(N)[::-1]


def _suite_legend(fig, ax, y=-0.090):
    handles = [Line2D([], [], ls="none", marker=SUITE_MARKER[s], mfc="none", mec=INK, mew=0.9,
                      ms=4.5, label=s) for s in SUITES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(SUITES),
              fontsize=FS["legend"], handletextpad=0.4, labelspacing=0.25, columnspacing=1.4,
              borderpad=0.0, frameon=False, labelcolor=INK)


def _xaxis(ax):
    ax.set_xlim(0.4, N + 0.6)
    ax.set_xticks(range(1, N + 1))
    ax.set_xlabel(f"mean rank across all {NDS} benchmark datasets  (1 = best of {N})")
    ax.grid(axis="x", ls=":", lw=0.5, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(Y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.62, N - 0.42)


def _row(ax, yi, a, ms_mean=7.0):
    c, r = ARMS[a]["color"], RANKS.loc[a]
    for s in SUITES:
        if np.isfinite(r[s]):
            ax.plot(r[s], yi, marker=SUITE_MARKER[s], mfc="none", mec=c, mew=0.9, ms=4.2,
                    ls="none", zorder=2)
    if np.isfinite(r.se_rank):
        ax.errorbar(r.mean_rank, yi, xerr=r.se_rank, fmt="none", ecolor=c, elinewidth=1.1,
                    capsize=STYLE["cap_size"], capthick=1.1, zorder=3)
    ax.plot(r.mean_rank, yi, marker="o", ms=ms_mean, color=c, mec="white", mew=0.8, zorder=4)
    return r


# ---------------------------------------------------------------- v2: compact ledger ---------
def build_v2():
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.30 * N + 0.85))
    yt = ax.get_yaxis_transform()
    for yi in Y:
        ax.axhline(yi, color=STYLE["faint"], lw=0.5, zorder=0)
    for yi, a in zip(Y, ORDER):
        r = _row(ax, yi, a)
        ax.text(-0.012, yi + 0.17, system(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], fontweight="bold", color=INK)
        ax.text(-0.012, yi - 0.17, label(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], color=INK)
        # values in ONE aligned column at the right — no numbers floating over the data
        ax.text(1.012, yi, f"{r.mean_rank:.1f}", transform=yt, ha="left", va="center",
                fontsize=FS["annot"], fontweight="bold", color=INK)
        ax.text(1.062, yi, f"± {r.se_rank:.1f}", transform=yt, ha="left", va="center",
                fontsize=FS["annot"], color=INK)
    _xaxis(ax)
    ax.text(1.012, N - 0.30, "mean", transform=yt, ha="left", va="center",
            fontsize=FS["annot"], fontweight="bold", color=INK)
    ax.text(1.062, N - 0.30, "± SE", transform=yt, ha="left", va="center",
            fontsize=FS["annot"], color=INK)
    _suite_legend(fig, ax, y=-0.105)
    title(ax, f"Mean rank across {NDS} benchmark datasets")
    return fig


# ---------------------------------------------------------------- v3: zebra, one line --------
def build_v3():
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.28 * N + 0.85))
    yt = ax.get_yaxis_transform()
    for k, yi in enumerate(Y):
        if k % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#F4F4F4", lw=0, zorder=0)
    for yi, a in zip(Y, ORDER):
        r = _row(ax, yi, a)
        # ONE right-aligned string: matplotlib cannot mix weights inside a text, and offsetting a
        # second text by an estimated width is what mangled the order here before.
        ax.text(-0.012, yi, f"{system(a)} · {label(a)}", transform=yt, ha="right", va="center",
                fontsize=FS["tick"], color=INK)
        ax.text(1.012, yi, f"{r.mean_rank:.1f}", transform=yt, ha="left", va="center",
                fontsize=FS["annot"], fontweight="bold", color=INK)
    _xaxis(ax)
    _suite_legend(fig, ax, y=-0.105)
    title(ax, f"Mean rank across {NDS} benchmark datasets")
    return fig


# ---------------------------------------------------------------- v4: split panel ------------
def build_v4():
    fig, axes = plt.subplots(1, 2, figsize=(STYLE["col2"], 0.30 * N + 0.9),
                             gridspec_kw=dict(width_ratios=[2.4, 1.0], wspace=0.06))
    ax, axs = axes
    yt = ax.get_yaxis_transform()
    for yi in Y:
        ax.axhline(yi, color=STYLE["faint"], lw=0.5, zorder=0)
        axs.axhline(yi, color=STYLE["faint"], lw=0.5, zorder=0)
    for yi, a in zip(Y, ORDER):
        c, r = ARMS[a]["color"], RANKS.loc[a]
        if np.isfinite(r.se_rank):
            ax.errorbar(r.mean_rank, yi, xerr=r.se_rank, fmt="none", ecolor=c, elinewidth=1.1,
                        capsize=STYLE["cap_size"], capthick=1.1, zorder=3)
        ax.plot(r.mean_rank, yi, marker="o", ms=7.0, color=c, mec="white", mew=0.8, zorder=4)
        for s in SUITES:                                   # suites live in their OWN panel
            if np.isfinite(r[s]):
                axs.plot(r[s], yi, marker=SUITE_MARKER[s], mfc="none", mec=c, mew=0.9, ms=4.2,
                         ls="none", zorder=2)
        ax.text(-0.012, yi + 0.17, system(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], fontweight="bold", color=INK)
        ax.text(-0.012, yi - 0.17, label(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], color=INK)
    _xaxis(ax)
    ax.set_xlabel(f"mean rank  (1 = best of {N})")
    axs.set_xlim(0.4, N + 0.6)
    axs.set_xticks([1, N // 2, N])
    axs.set_xlabel("per-suite rank")
    axs.grid(axis="x", ls=":", lw=0.5, color=STYLE["grid"])
    axs.set_axisbelow(True)
    axs.set_yticks(Y)
    axs.set_yticklabels([])
    axs.set_ylim(-0.62, N - 0.42)
    axs.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axs.spines[sp].set_visible(False)
    _suite_legend(fig, axs, y=-0.105)
    title(fig, f"Mean rank across {NDS} benchmark datasets, and its spread across the four suites")
    return fig


# ---------------------------------------------------------------- v5: lollipop ---------------
def build_v5():
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.30 * N + 0.85))
    yt = ax.get_yaxis_transform()
    for yi, a in zip(Y, ORDER):
        c, r = ARMS[a]["color"], RANKS.loc[a]
        ax.plot([0.4, r.mean_rank], [yi, yi], color=c, lw=1.4, alpha=0.55, zorder=1,
                solid_capstyle="butt")
        r = _row(ax, yi, a, ms_mean=7.5)
        ax.text(-0.012, yi + 0.17, system(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], fontweight="bold", color=INK)
        ax.text(-0.012, yi - 0.17, label(a), transform=yt, ha="right", va="center",
                fontsize=FS["tick"], color=INK)
        ax.text(1.012, yi, f"{r.mean_rank:.1f}", transform=yt, ha="left", va="center",
                fontsize=FS["annot"], fontweight="bold", color=INK)
    _xaxis(ax)
    _suite_legend(fig, ax, y=-0.105)
    title(ax, f"Mean rank across {NDS} benchmark datasets")
    return fig


def main():
    for name, fn in [("figA1_v1", build_v1), ("figA1_v2", build_v2), ("figA1_v3", build_v3),
                     ("figA1_v4", build_v4), ("figA1_v5", build_v5)]:
        fig = fn()
        save(fig, name)
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Fig A1 — overall standing across every benchmark we ran.

ONE script, ONE figure: figures_v2/figA1.png / .pdf

What it shows
-------------
Each model is ranked within every individual dataset (1 = best of N), and those ranks are averaged
over all 66 datasets: MoleculeNet (7) · MoleculeACE (30 ChEMBL targets) · Polaris (28 ADMET/kinase
tasks, each on its own primary metric) · CBS (1). Ranking per dataset is what makes the pooling
legal — the metrics are heterogeneous (RMSE, ROC-AUC, NEF1%, Pearson r) and cannot be averaged.
A dataset scored on only k of the N models is rescaled from [1..k] to [1..N] so a missing model
cannot flatter the rest.

  filled dot    mean rank over all datasets
  bar           ±1 SE, corrected for the design effect (see below)
  open markers  the four per-suite mean ranks

Why the SE is corrected
-----------------------
Datasets inside a suite largely agree about which model is better — MoleculeACE's 30 targets
correlate at rho = 0.74 and behave like ~1.3 independent datasets. Treating all 66 as independent
would understate the SE by ~3x, so it is inflated by sqrt(design effect) in allsuites.wide_ranks().
The ordering itself is robust: Kendall tau 0.78-1.00 against per-suite weighting and against every
leave-one-suite-out variant (scripts/weighting_sensitivity.py).

Caption text is NOT drawn into the figure — it goes in the LaTeX \\caption{}. Use the paragraphs
above as its source.

Run:  python3 -m figures.fig_A1
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, ARM_ORDER, system, label
from figures.allsuites import wide_ranks, wide_table, SUITES

check_font()

SUITE_MARKER = {"MoleculeNet": "o", "MoleculeACE": "^", "Polaris": "s", "CBS": "D"}
INK = "#000000"
BAND = "#F2F2F2"          # zebra row band; light enough not to compete with the marker colours

# Arms are registered in arms.py before their GPU results land. Plot only arms with essentially
# COMPLETE coverage (>=60 of 66 datasets): a mean rank computed on a sliver of the suite is not
# comparable to the 66/66 mainline arms (user decision 2026-08-17: A1 stays as approved -- the 16
# mainline arms). Today this excludes: chemeleon_e2e (structurally <=8/66 in A1's sources -- no
# MoleculeACE/Polaris exists for it; the e2e comparator lives in Fig A2, chemeleon_frozen remains
# the CheMeleon row here) and s2u_dense (31/66 -- MoleculeACE + CBS landed 2026-08-17, its
# MolNet 5-fold CV and Polaris are still pending; it auto-enters at >=60).
_S0, _ = wide_table(ARM_ORDER)
ARMS_USED = [a for a in ARM_ORDER if _S0.loc[a].notna().sum() >= 60]
N = len(ARMS_USED)

RANKS, PER_DATASET, META = wide_ranks(ARMS_USED, per_suite_equal=False)
NDS = int(PER_DATASET.notna().sum(axis=1).max())


def suite_handles():
    """The four suite markers -- shared by build() and the assembled figures/fig_A.py."""
    return [Line2D([], [], ls="none", marker=SUITE_MARKER[s], mfc="none", mec=INK, mew=0.9,
                   ms=4.5, label=s) for s in SUITES]


def draw(ax, compact=False):
    """Render the ranking into a supplied axes. `compact` trims the per-point value labels and the
    marker sizes for the assembled fig_A layout, where this panel is much narrower than standalone.
    Used by both build() and figures/fig_A.py — the drawing lives in ONE place."""
    order = list(RANKS.index)
    y = np.arange(N)[::-1]
    ytrans = ax.get_yaxis_transform()          # x in axes coords, y in data coords

    # Alternating background bands instead of a hairline per row: they carry the eye across the
    # full width to the per-suite markers without adding another line to the plot. Drawn under
    # everything (zorder 0) and clipped to the row pitch so the top/bottom rows are not clipped
    # by the axes limits.
    for k, yi in enumerate(y):
        if k % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)

    for yi, a in zip(y, order):
        c, r = ARMS[a]["color"], RANKS.loc[a]
        for s in SUITES:                                   # per-suite mean ranks
            if np.isfinite(r[s]):
                ax.plot(r[s], yi, marker=SUITE_MARKER[s], mfc="none", mec=c, mew=0.9, ms=(3.6 if compact else 4.6),
                        ls="none", zorder=2)
        if np.isfinite(r.se_rank):
            ax.errorbar(r.mean_rank, yi, xerr=r.se_rank, fmt="none", ecolor=c, elinewidth=1.1,
                        capsize=STYLE["cap_size"], capthick=1.1, zorder=3)
        ax.plot(r.mean_rank, yi, marker="o", ms=(5.6 if compact else 7.5), color=c, mec="white", mew=0.8, zorder=4)
        if not compact:
            ax.text(r.mean_rank, yi + 0.30, f"{r.mean_rank:.1f}", ha="center", va="bottom",
                    fontsize=FS["annot"], color=INK)
        # two-line row label: system in bold, recipe below in regular. Drawn by hand rather than
        # via tick labels because matplotlib cannot mix weights inside one tick string.
        ax.text(-0.012, yi + 0.19, system(a), transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"], fontweight="bold", color=INK)
        ax.text(-0.012, yi - 0.19, label(a), transform=ytrans, ha="right", va="center",
                fontsize=FS["tick"], color=INK)

    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_ylim(-0.62, N - 0.42)
    ax.set_xlim(0.4, N + 0.6); ax.set_xticks(range(1, N + 1, 3) if compact else range(1, N + 1))
    ax.set_xlabel(f"mean rank ({NDS} datasets)" if compact else
                  f"mean rank across all {NDS} benchmark datasets  (1 = best of {N})")
    ax.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"]); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if not compact:
        ax.legend(handles=suite_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.090),
                  ncol=len(SUITES), fontsize=FS["legend"], handletextpad=0.4, labelspacing=0.25,
                  columnspacing=1.4, borderpad=0.0, frameon=False, labelcolor=INK)


def build():
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.38 * N + 0.75))
    draw(ax)
    title(ax, f"Overall standing across every benchmark ({NDS} datasets, 4 suites)")
    return fig


def main():
    print(f"Fig A1 · mean rank over {NDS} datasets (1 = best of {N})\n")
    print(f"{'model':38s} {'mean':>6s} {'±SE':>5s} | " + " ".join(f"{s[:7]:>7s}" for s in SUITES))
    for a in RANKS.index:
        r = RANKS.loc[a]
        per = " ".join(f"{r[s]:7.2f}" if np.isfinite(r[s]) else "      —" for s in SUITES)
        print(f"{system(a) + ' / ' + label(a):38s} {r.mean_rank:6.2f} {r.se_rank:5.2f} | {per}")
    print()
    fig = build()
    save(fig, "fig_A1", subdir="panels")
    plt.close(fig)


if __name__ == "__main__":
    main()

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
  †             the whole network is FINE-TUNED on each task; every other arm is a FROZEN
                encoder with a trained probe

READ THE DAGGER BEFORE THE ORDERING. Two of the 18 arms fine-tune end-to-end and 16 do not, and on
this benchmark set that difference outweighs any pretraining difference. CheMeleon gains +0.173
macro RMSE on MoleculeACE by fine-tuning (0.8256 frozen -> 0.6526 e2e, 21%); CLIMB gains 0.010-0.015
(1-2%) from the same treatment. A message-passing network fine-tunes far better than a transformer
used as a frozen probe, so `CheMeleon (e2e)` topping the table is a statement about FINE-TUNING,
not about pretraining quality.
The two CheMeleon rows are the control for exactly this, and they sit 7 rank positions apart:
frozen-vs-frozen, CheMeleon is 9.80 against CLIMB supervised-desc 6.20 and unsupervised 8.09 --
i.e. CLIMB's REPRESENTATION is the better one, and CheMeleon overtakes it only when allowed to
retrain its encoder. Restricted to the 16 frozen arms the ranking is ECFP+desc 2.12, ECFP 4.48,
supervised-desc 5.21, CheMeleon-frozen 8.56.

REPLICATION. Every CLIMB arm is 3 pretraining seeds x 3 eval seeds. ECFP, ECFP+desc and the two
CheMeleon variants have ONE pretraining seed by construction -- a deterministic featurizer and a
fixed external model have no pretraining stage to replicate -- but they are NOT unreplicated: their
head / fine-tuning seed is replicated 3x inside the single directory (CheMeleon e2e on MoleculeACE:
0.6503 / 0.6547 / 0.6526, sd 0.0022; frozen: 0.8377 / 0.8212 / 0.8180, sd 0.0106). A
directory-counting seed check reports "1" for these and is wrong to; audit check 3 now requires
>= 3 replicate CELLS from them instead.

Why the SE is corrected
-----------------------
Datasets inside a suite largely agree about which model is better — MoleculeACE's 30 targets
correlate at rho = 0.74 and behave like ~1.3 independent datasets. Treating all 66 as independent
would understate the SE by ~3x, so it is inflated by sqrt(design effect) in allsuites.wide_ranks().
The ordering is broadly robust (Kendall tau 0.843 against per-suite weighting) BUT THE TOP POSITION
IS NOT, and that must not be quoted as if it were. Weighting every dataset equally lets MoleculeACE
(30) and Polaris (28) decide 58 of 66; weighting the four SUITES equally instead gives
ECFP+desc 2.68 and CheMeleon (e2e) 4.78 -- the order flips. ECFP+desc in fact wins 2 of the 4
suites outright (MoleculeNet 4.14 vs 7.41, CBS 1.00 vs 8.00) and loses the two large ones.
So "CheMeleon (e2e) is first overall" is an artefact of TWO choices stacking -- per-dataset
weighting and the frozen-vs-fine-tuned protocol above -- and the defensible claim is that
ECFP+desc is the best model here, with CheMeleon competitive on MoleculeACE and Polaris when it is
allowed to fine-tune.

CONSISTENT WITH THE CHEMELEON PAPER, and worth saying so explicitly (Burns et al.,
arXiv:2506.15792v2). They evaluate on Polaris + MoleculeACE -- 58 datasets, which is EXACTLY the
two suites where we also find CheMeleon ahead, and which supply 58 of our 66 under per-dataset
weighting. Their reported Polaris win rate is CheMeleon 75% (21/28) against Random Forest 68%
(19/28), where a "win" counts being best OR statistically indistinguishable from best (Tukey HSD),
i.e. a two-benchmark margin over the classical baseline rather than a decisive one. Their headline
also uses END-TO-END fine-tuning, matching what we find drives the gap.
Two things we add rather than contradict: the suites they did not test (MoleculeNet, CBS) are the
ones where the classical anchor wins outright, and their classical baseline is Random Forest, not
XGBoost on ECFP+descriptors.
NOTE ALSO what CheMeleon IS: a D-MPNN pretrained to predict 1613 Mordred descriptors on 1M PubChem
molecules. That is our `supervised, desc` objective in a different architecture -- so it belongs in
fig_E's supervised panel conceptually, and our descriptor-residual finding (a descriptor-pretrained
encoder is 81-88% linearly explained by the descriptors it was trained on, and adds nothing on top
of ECFP+desc in fig_F) is a direct comment on its premise, not an unrelated result.

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
# comparable to the 66/66 mainline arms (user decision 2026-08-17: A1 stays as approved -- the
# mainline arms). Coverage as of 2026-08-18, after chemeleon_e2e's full 28-task Polaris run landed:
#   66/66  every arm in ARM_ORDER, including chemeleon_e2e (was 31/66, then 39/66 once two loader
#          bugs in allsuites were fixed, now complete) and s2u_dense (was 31/66).
# The frozen-vs-e2e SPLIT IS GONE. Panel (a) previously showed CheMeleon FROZEN while panel (b)
# showed CheMeleon E2E -- different models 65 kcal/mol apart on QM7 under one comparator name --
# because only the frozen variant cleared the coverage floor. Both now qualify and both are drawn,
# so fig_A2.short() naming the probe in the legend is now a convenience rather than a correction.
# NOTE the headline consequence: chemeleon_e2e enters at mean rank 2.53, AHEAD of the ECFP+desc
# anchor at 2.87. The classical anchor is no longer first overall.
# PROBE PROTOCOL. 16 of the 18 arms are FROZEN encoders with a trained probe; two fine-tune the
# whole network on each downstream task, and they are marked with a dagger because that difference
# is worth more than any pretraining difference in this figure. CheMeleon gains +0.173 macro RMSE
# on MoleculeACE from fine-tuning (0.8256 -> 0.6526, 21%) where CLIMB gains 0.010-0.015 (1-2%) --
# a message-passing network fine-tunes far better than a frozen-probe transformer. So CheMeleon's
# top rank is a statement about FINE-TUNING, not about pretraining quality, and the two CheMeleon
# rows are 7 rank positions apart for exactly that reason. Read them together: frozen-vs-frozen,
# CheMeleon is 9.80 against supervised-desc 6.20 and unsupervised 8.09.
E2E_ARMS = {"chemeleon_e2e", "e2e_no_pretrain"}

_S0, _ = wide_table(ARM_ORDER)
ARMS_USED = [a for a in ARM_ORDER if _S0.loc[a].notna().sum() >= 60]
N = len(ARMS_USED)

# PER-SUITE EQUAL WEIGHTING (user decision 2026-08-19). Each arm's headline number is the mean of
# its FOUR SUITE mean-ranks, not the mean over all 66 datasets. Per-dataset weighting lets
# MoleculeACE (30) and Polaris (28) decide 58 of 66, which is exactly the two suites CheMeleon was
# built and tuned against (Burns et al. evaluate on those two only) -- so it inflates an arm that
# is strong there and absent-to-weak on MoleculeNet (7) and CBS (1). Under equal suite weight the
# order flips back: ECFP+desc 2.68 first, CheMeleon (e2e) 4.53 second. The per-suite open markers
# already drawn on every row let a reader verify the aggregation by eye.
RANKS, PER_DATASET, META = wide_ranks(ARMS_USED, per_suite_equal=True)
NDS = int(PER_DATASET.notna().sum(axis=1).max())


def suite_handles(with_dagger=False):
    """The four suite markers -- shared by build() and the assembled figures/fig_A.py.

    The dagger key is NOT here: a 5th entry with no marker and a sentence-length label wrecks the
    4-column suite row, so draw() writes it as a line of text under the axes instead. The symbol
    was carrying the single most
    important caveat in this figure -- that two arms fine-tune the whole network while sixteen use
    a frozen encoder -- with nothing on the canvas to decode it (user 2026-08-19: "Chemeleon e2e
    also has a cross which I don't understand"). An unexplained mark next to the arm that places
    second is worse than no mark.
    """
    del with_dagger          # kept for call-site compatibility; the note is drawn by draw()
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
        # The number goes on EVERY row in both modes (user 2026-08-19). It was dropped in compact
        # to save width, but the assembled fig_A is where most readers meet this panel, and a rank
        # plot whose ranks must be estimated off a tick axis is not a ranking. In compact the label
        # sits to the RIGHT of the dot rather than above it, clear of the SE bar and of the row
        # above, which is what the vertical room was actually being spent on.
        if compact:
            # PAST THE RIGHTMOST MARK ON THIS ROW, not past the whisker. The four open suite
            # markers routinely sit outside the SE bar -- CBS is a single dataset, so its rank
            # swings far from the mean -- and anchoring to mean+se dropped the label straight on
            # top of them ("2.7" over ECFP+desc's Polaris square, "14.6" over MiniMol's triangle).
            rightmost = max([r.mean_rank + r.se_rank] +
                            [r[s_] for s_ in SUITES if np.isfinite(r[s_])])
            ax.text(rightmost + 0.55, yi, f"{r.mean_rank:.1f}", ha="left",
                    va="center", fontsize=FS["annot"] - 1, color=INK, zorder=5)
        else:
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
    ax.set_xlim(0.4, N + (3.1 if compact else 0.6)); ax.set_xticks(range(1, N + 1, 3) if compact else range(1, N + 1))
    ax.set_xlabel(f"mean rank, suites equally weighted" if compact else
                  f"mean of the 4 suite mean-ranks  ({NDS} datasets, 1 = best of {N})")
    ax.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"]); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # NO DAGGER AND NO FOOTNOTE (user 2026-08-19). The frozen-vs-fine-tuned split is still the most
    # important caveat in this figure -- see the module docstring, which carries the numbers -- but
    # it belongs in the LaTeX caption, not as a mark on the plot. E2E_ARMS is kept because the
    # docstring and the caption are written from it.

    if not compact:
        ax.legend(handles=suite_handles(), loc="upper center",
                  bbox_to_anchor=(0.5, -0.095), ncol=len(SUITES), fontsize=FS["legend"], handletextpad=0.4, labelspacing=0.25,
                  columnspacing=1.4, borderpad=0.0, frameon=False, labelcolor=INK)


def build():
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 0.38 * N + 0.75))
    draw(ax)
    title(ax, f"Overall standing: mean of the 4 suite ranks ({NDS} datasets)")
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

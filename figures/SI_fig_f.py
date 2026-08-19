"""SI Fig f — what can each representation actually RESOLVE?

ONE script, ONE figure: figures_v2/SI_fig_f.png / .pdf

Thirteen chemical failure modes, 100 molecule PAIRS each, scored in two OPPOSITE directions:

  CLASS A   the two molecules are genuinely DIFFERENT   -> success = the embedding SEPARATES them
  CLASS B   the two SMILES denote the SAME molecule     -> success = it does NOT separate them

The axis therefore means opposite things in the two blocks, which is why they are drawn as two
labelled blocks with the class B panels tinted rather than as one 13-panel grid. A reader who
misses the flip reads the figure exactly backwards.

TWO QUESTIONS, ONE GRID
-----------------------
"Is the difference THERE" and "how BIG is it" are different questions, and conflating them cost
this figure two rebuilds. The metric is now the first question, with no threshold at all: RESOLVED
means the two embedding vectors DIFFER, guarded only at float32 epsilon (cosine > 1e-6). That guard
is arithmetic, not chemistry -- re-embedding the same molecules in a different batch order
reproduces every representation BIT-FOR-BIT, 128/128, largest cosine 1.8e-07, so the guard sits two
orders above measured noise (scripts/resolution_noise_floor.py). For a deterministic fingerprint it
collapses to bit-equality.

The BARS answer the second question -- median separation, log axis -- because that is where the
interesting structure is, and only 13 of the 65 cells are not at 100% on the first. Those 13 are
annotated with their rate, so the sparse binary answer rides along instead of costing a second
13-panel grid.

WHAT THIS COST, RECORDED BECAUSE IT WAS A REAL RETRACTION. An earlier version thresholded the
separation at eps=0.01 and reported "CLIMB sup resolves 1% of inverted stereocentres, so the CLMs
cannot see stereochemistry". That does not survive: under the threshold-free definition CLIMB sup
resolves 100% of them. It DOES produce a different vector -- by 8.6e-5 against ECFP4's 9.0e-2. The
information is present and about a thousandfold compressed, which is a weaker and more accurate
claim than blindness, and it is the bars rather than the rate that carry it.

Success is NOT bit-equality — that is meaningless for a continuous embedding, and it was the peer
session's first cut: CheMeleon reads 0% identical on class B purely from floating-point summation
order while its true separation is exactly zero. It is a scale-free SEPARATION RATIO instead:
cos(A,B) over the median cos(A, 1000 random molecules), thresholded at eps=0.01, i.e. "at least 1%
of the way to a random molecule". success_rates.csv carries eps=0.001/0.01/0.05 and the ranking
does not depend on which is used.

WHAT THE FIGURE SAYS
--------------------
1. THE CLMs COMPRESS STEREOCHEMISTRY ALMOST TO NOTHING. They do separate enantiomers -- 100% under
   the threshold-free definition -- but by 8.6e-5 (CLIMB sup) and 3.7e-3 (CLIMB unsup) against
   ECFP4+stereo's 9.0e-2: three orders of magnitude. A configuration flip is a textbook activity
   cliff, so a representation that encodes it a thousandfold more weakly than a fingerprint is
   unlikely to support it downstream. This is a statement about MAGNITUDE and must be quoted from
   the bars, never as a rate.
2. THE CLMs ARE NOT INVARIANT TO HOW A MOLECULE IS WRITTEN -- the most robust result here, stable
   under every definition tried. Class B: 0% on re-written SMILES and 0% on Kekule forms. Same molecule, same graph, different string, different embedding -- for Kekule
   the median separation is 4.4x the distance to a RANDOM molecule. Fingerprints and CheMeleon are
   invariant by construction. The untrained control is equally bad (0-1%), so this is a property of
   tokenized SMILES, not of our pretraining.
3. THE BLIND SPOTS ARE COMPLEMENTARY, which is why all 13 panels are shown rather than a summary.
   ECFP4 is blind to isotopes (4%) exactly where the CLMs are perfect (97-100%), because a [13C]
   token changes the string while Morgan atom invariants ignore isotope. CheMeleon is blind to
   isotopes AND stereochemistry (0%) but is the best of the five at ring size (89%). No
   representation dominates; each fails somewhere specific.

`matched_mw` and `matched_descriptors` are 100% everywhere and are kept precisely BECAUSE they are
flat: "the embedding only encodes bulk properties" is the natural objection to (1) and (2), and
these two panels are the control that refutes it.

CONTROLS ARE NOT DRAWN. `random encoder` and `ECFP4 stereo-blind` are in the CSV and belong in the
caption: the stereo-blind row scores 0% on stereo_flip and ez_flip BY CONSTRUCTION, which is what
validates the harness, and the untrained encoder is the comparator for claim (1).

Data: figure_data/embedding_resolution/success_rates.csv (peer session, commit ad8fbc6); pairs,
per-pair distances, the raw vectors and MCS-aligned depictions of 8 pairs per mode are beside it.

Run:  python3 -m figures.SI_fig_f
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# The y axis is "success", not "resolved": in class B a HIGH bar means the representation correctly
# did NOT separate two spellings of one molecule. Labelling it "resolved" would make the class B
# block read as the opposite of what it shows -- the one mistake this figure's layout exists to
# prevent. The direction lives in the block labels, so the axis stays neutral.

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS

check_font()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "embedding_resolution" / "success_rates.csv"
INK = "#000000"
TINT = "#F0EDE6"          # class B panel background; warm, so it reads as "different rule"

# CSV label -> (arms.py key for the colour, legend label). The five drawn representations, in the
# order the peer specified. Controls are deliberately absent -- see the docstring.
SERIES = [("ECFP4+stereo",     "ecfp",             "ECFP4+stereo"),
          ("ECFP4+desc",       "ecfp_desc",        "ECFP4+desc"),
          ("CLIMB sup",        "sup_dense",        "CLIMB sup"),
          ("CLIMB unsup",      "unsup",            "CLIMB unsup"),
          ("CheMeleon",        "chemeleon_frozen", "CheMeleon")]

# (class, mode, two-line panel title). Order is the peer's; the class blocks are drawn separately.
MODES = [("A", "stereo_flip",         "Inverted\nstereocentre"),
         ("A", "ez_flip",             "Flipped E/Z\ndouble bond"),
         ("A", "c_to_n",              "Aromatic C→N\n(benzene→pyridine)"),
         ("A", "add_methyl",          "One methyl\nadded"),
         ("A", "add_fluorine",        "One fluorine\nadded"),
         ("A", "isotope_13c",         "$^{12}$C→$^{13}$C,\ngraph unchanged"),
         ("A", "ring_size",           "Cyclopentyl ↔\ncyclohexyl"),
         ("A", "regioisomer",         "ortho vs meta\nsubstitution"),
         ("A", "matched_mw",          "Different molecules,\nsame MW"),
         ("A", "matched_descriptors", "Different molecules,\nsame descriptors"),
         ("B", "smiles_enumeration",  "Re-written\nSMILES"),
         ("B", "kekule",              "Kekulé\nform"),
         ("B", "symmetry_equivalent", "Equivalent\npositions")]
NCOL = 5


def compute():
    """(median separation, binary success %). Both threshold-free; see the module docstring."""
    d = pd.read_csv(SRC)
    return (d.pivot_table(index=["klass", "mode"], columns="embedding", values="median_separation"),
            d.pivot_table(index=["klass", "mode"], columns="embedding", values="success"))


FLOOR = 3e-6          # log-axis bottom; exact zeros are drawn AT it with an open marker


def _panel(ax, vals, rates, title, klass):
    """One mode. Bars are median separation on a LOG axis; the three eps cuts are reference lines.

    An EXACT zero is not a small number -- for a fingerprint on class B it means invariant by
    construction -- so it cannot share the log axis with 1e-5. Those are drawn as an open marker
    sitting on the floor, which reads as "off the bottom of this scale" rather than as a short bar.
    """
    if klass == "B":
        ax.set_facecolor(TINT)
    x = np.arange(len(SERIES))
    v = np.asarray(vals, dtype=float)
    drawn = np.where(v > 0, v, FLOOR)
    ax.bar(x[v > 0], drawn[v > 0], width=0.74,
           color=[ARMS[k]["color"] for (_, k, _), keep in zip(SERIES, v > 0) if keep],
           edgecolor=INK, linewidth=0.6, zorder=3, bottom=FLOOR)
    for xi, (_, k, _) in zip(x[v == 0], [S for S, z in zip(SERIES, v == 0) if z]):
        ax.plot(xi, FLOOR * 2.2, marker="o", ms=3.4, mfc="white", mec=ARMS[k]["color"],
                mew=1.1, ls="none", zorder=4, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 20)
    ax.set_yticks([1e-5, 1e-3, 1e-1])
    ax.set_yticklabels(["$10^{-5}$", "$10^{-3}$", "$10^{-1}$"])
    ax.set_xticks([])
    ax.set_xlim(-0.72, len(SERIES) - 0.28)
    ax.grid(axis="y", ls=":", lw=0.5, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    # The BINARY rate rides along only where it is not 100%: 13 of 65 cells. Printing it on all 65
    # would bury the 13 that say something, and giving it its own 13-panel grid would double the
    # figure to carry a mostly-constant column.
    for xi, r, hv in zip(x, rates, drawn):
        if np.isfinite(r) and r < 99.5:
            ax.text(xi, hv * 1.9 if hv > FLOOR * 3 else FLOOR * 9, f"{r:.0f}%", ha="center",
                    va="bottom", fontsize=FS["annot"] - 2, color=INK, zorder=6)
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def main():
    P, R = compute()
    fig = plt.figure(figsize=(STYLE["col2"], 5.15))
    # Two BLOCKS, not one grid: the y axis means opposite things in each, so they are separated by
    # a real gap and labelled on the left. Class A is 10 panels (2 rows of 5); class B is 3.
    gsA = fig.add_gridspec(2, NCOL, left=0.085, right=0.995, top=0.915, bottom=0.435,
                           wspace=0.34, hspace=0.62)
    gsB = fig.add_gridspec(1, NCOL, left=0.085, right=0.995, top=0.315, bottom=0.145,
                           wspace=0.34)
    A = [m for m in MODES if m[0] == "A"]
    B = [m for m in MODES if m[0] == "B"]
    missing = []
    for i, (kl, mode, title) in enumerate(A):
        ax = fig.add_subplot(gsA[i // NCOL, i % NCOL])
        vals = [P.loc[(kl, mode), lab] if (kl, mode) in P.index else np.nan for lab, _, _ in SERIES]
        rates = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan for lab, _, _ in SERIES]
        if not np.isfinite(vals).any():
            missing.append(mode)
        _panel(ax, vals, rates, title, kl)
        if i % NCOL == 0:
            ax.set_ylabel("median separation", fontsize=FS["annot"])
    for i, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gsB[0, i])
        vals = [P.loc[(kl, mode), lab] if (kl, mode) in P.index else np.nan for lab, _, _ in SERIES]
        rates = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan for lab, _, _ in SERIES]
        if not np.isfinite(vals).any():
            missing.append(mode)
        _panel(ax, vals, rates, title, kl)
        if i == 0:
            ax.set_ylabel("median separation", fontsize=FS["annot"])
    assert not missing, f"SI fig f: no data for {missing}"

    # the two block labels carry the DIRECTION, which is the thing a reader must not miss
    fig.text(0.012, (0.915 + 0.435) / 2, "Different molecules\nmust SEPARATE", rotation=90,
             va="center", ha="center", fontsize=FS["annot"], fontweight="bold", color=INK)
    fig.text(0.012, (0.315 + 0.145) / 2, "Same molecule\nmust NOT separate", rotation=90,
             va="center", ha="center", fontsize=FS["annot"], fontweight="bold", color=INK)

    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=ARMS[k]["color"], edgecolor=INK, lw=0.7, label=lab)
               for _, k, lab in SERIES]
    # the two glyphs a reader cannot guess: an EXACT zero is not a small bar, and the dashed lines
    # are where a success-rate threshold would cut -- which is how this figure shows that four
    # modes are knife-edge without needing a footnote to say so
    handles += [Line2D([], [], marker="o", ms=3.6, mfc="white", mec=INK, mew=1.1, ls="none",
                       label="exactly 0 (invariant)"),
                Line2D([], [], ls="none", label="")]     # spacer; keeps the 3x2 legend balanced
    handles = [h for h in handles if h.get_label()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.090), ncol=6,
               fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.5, labelspacing=0.35,
               borderpad=0.0, frameon=False)
    save(fig, "SI_fig_f")
    plt.close(fig)

    print("\nSI Fig f — median separation, with the binary resolved-rate where it is not 100%\n")
    print(f"   {'class':<6}{'mode':<22}" + "".join(f"{lab:>14s}" for lab, _, _ in SERIES))
    for kl, mode, _ in MODES:
        row = f"   {kl:<6}{mode:<22}"
        for lab, _, _ in SERIES:
            v = P.loc[(kl, mode), lab] if (kl, mode) in P.index else np.nan
            row += (f"{v:>14.2e}" if np.isfinite(v) else f"{'—':>14}")
        print(row)
    d = pd.read_csv(SRC)
    e = d[d.embedding.isin([lab for lab, _, _ in SERIES])].copy()
    # The metric is threshold-free now, so there is no eps sensitivity left to report. What IS
    # worth printing is where the binary rate says something the bars do not.
    nb = e[e.success < 99.5]
    print(f"\n   BINARY RESOLVED-RATE below 100% in {len(nb)} of {len(e)} cells (these are the "
          f"annotated bars):")
    for _, r in nb.sort_values("success").iterrows():
        print(f"     {r.klass} {r['mode']:<20} {r.embedding:<14} {r.success:5.0f}%   "
              f"median separation {r.median_separation:.2e}")
    ctl = d[d.embedding.isin(["random encoder", "ECFP4 stereo-blind"])]
    print("\n   CONTROLS (not drawn; caption material):")
    for emb, g in ctl.groupby("embedding"):
        g = g.set_index("mode")
        bits = "  ".join(f"{m}={g.loc[m,'success']:.0f}%" for m in
                         ("stereo_flip", "ez_flip", "kekule") if m in g.index)
        print(f"     {emb:<22}{bits}")


if __name__ == "__main__":
    main()

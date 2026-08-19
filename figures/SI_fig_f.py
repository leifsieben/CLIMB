"""SI Fig f — what can each representation actually RESOLVE?

ONE script, ONE figure: figures_v2/SI_fig_f.png / .pdf

Thirteen chemical failure modes, 100 molecule PAIRS each, scored in two OPPOSITE directions:

  CLASS A   the two molecules are genuinely DIFFERENT   -> success = the embedding SEPARATES them
  CLASS B   the two SMILES denote the SAME molecule     -> success = it does NOT separate them

The axis therefore means opposite things in the two blocks, which is why they are drawn as two
labelled blocks with the class B panels tinted rather than as one 13-panel grid. A reader who
misses the flip reads the figure exactly backwards.

WHY THIS PLOTS SEPARATION AND NOT A SUCCESS RATE
------------------------------------------------
The first version drew success % at eps=0.01 and it was not defensible: 19 of the 65 drawn cells
move by MORE THAN 20 POINTS across eps=0.001..0.05, and four modes invert outright. The worst case
carried a headline -- CLIMB unsup on inverted stereocentres reads 94% at eps=0.001, BEATING
ECFP4+stereo's 89%, and 4% at eps=0.01. "CLIMB unsup resolves 4% of stereocentres" was an artefact
of where the line was drawn, and no caption caveat rescues a bar chart with that property.

So the figure plots the underlying quantity -- MEDIAN SEPARATION, log axis -- and draws the three
eps thresholds as reference lines. A reader sees the magnitude AND how close it sits to any cut,
so a knife-edge cell is knife-edge by inspection rather than by footnote. The quantity spans five
orders of magnitude (8.6e-5 to 4.4), which is precisely why thresholding it lost information.

Success is NOT bit-equality — that is meaningless for a continuous embedding, and it was the peer
session's first cut: CheMeleon reads 0% identical on class B purely from floating-point summation
order while its true separation is exactly zero. It is a scale-free SEPARATION RATIO instead:
cos(A,B) over the median cos(A, 1000 random molecules), thresholded at eps=0.01, i.e. "at least 1%
of the way to a random molecule". success_rates.csv carries eps=0.001/0.01/0.05 and the ranking
does not depend on which is used.

WHAT THE FIGURE SAYS
--------------------
1. THE CLMs CANNOT SEE STEREOCHEMISTRY, THOUGH THEY READ IT. CLIMB sup resolves 1% of inverted
   stereocentres and 2% of E/Z flips. The tokenizer carries 13 '@' tokens and 20.8% of the corpus
   is stereo-bearing, so the information is in the input and does not survive into the embedding.
   The UNTRAINED control scores 84% and 89% on those same modes -- an untrained network separates
   stereoisomers far better than a pretrained one, so pretraining actively destroys this.
2. THE CLMs ARE NOT INVARIANT TO HOW A MOLECULE IS WRITTEN. Class B: 2% and 0% on re-written SMILES
   and Kekule forms. Same molecule, same graph, different string, different embedding -- for Kekule
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
    d = pd.read_csv(SRC)
    return d.pivot_table(index=["klass", "mode"], columns="embedding", values="median_separation")


FLOOR = 3e-6          # log-axis bottom; exact zeros are drawn AT it with an open marker
EPS = (0.001, 0.01, 0.05)


def _panel(ax, vals, title, klass):
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
    for e in EPS:
        ax.axhline(e, color=STYLE["ink"], ls=(0, (2, 2)), lw=0.55, zorder=2)
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
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def main():
    P = compute()
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
        if not np.isfinite(vals).any():
            missing.append(mode)
        _panel(ax, vals, title, kl)
        if i % NCOL == 0:
            ax.set_ylabel("median separation", fontsize=FS["annot"])
    for i, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gsB[0, i])
        vals = [P.loc[(kl, mode), lab] if (kl, mode) in P.index else np.nan for lab, _, _ in SERIES]
        if not np.isfinite(vals).any():
            missing.append(mode)
        _panel(ax, vals, title, kl)
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
                Line2D([], [], color=INK, ls=(0, (2, 2)), lw=0.55,
                       label="success cut-offs (ε = 0.001, 0.01, 0.05)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.098), ncol=4,
               fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.5, labelspacing=0.35,
               borderpad=0.0, frameon=False)
    save(fig, "SI_fig_f")
    plt.close(fig)

    print("\nSI Fig f — median separation (threshold-free; eps cuts at 0.001/0.01/0.05)\n")
    print(f"   {'class':<6}{'mode':<22}" + "".join(f"{lab:>14s}" for lab, _, _ in SERIES))
    for kl, mode, _ in MODES:
        row = f"   {kl:<6}{mode:<22}"
        for lab, _, _ in SERIES:
            v = P.loc[(kl, mode), lab] if (kl, mode) in P.index else np.nan
            row += (f"{v:>14.2e}" if np.isfinite(v) else f"{'—':>14}")
        print(row)
    d = pd.read_csv(SRC)
    e = d[d.embedding.isin([lab for lab, _, _ in SERIES])].copy()
    cols = ["success_eps0.001", "success_eps0.01", "success_eps0.05"]
    e["span"] = e[cols].max(axis=1) - e[cols].min(axis=1)
    bad = e[e.span > 20]
    print(f"\n   THRESHOLD SENSITIVITY: {len(bad)} of {len(e)} cells move >20 points across eps.")
    print("   Modes STABLE at every eps (safe to quote a success rate for):")
    for m in sorted(set(e["mode"]) - set(bad["mode"])):
        print(f"     {m}")
    print("   Modes that are NOT (quote median separation instead):")
    for m in sorted(set(bad["mode"])):
        print(f"     {m}")
    ctl = d[d.embedding.isin(["random encoder", "ECFP4 stereo-blind"])]
    print("\n   CONTROLS (not drawn; caption material):")
    for emb, g in ctl.groupby("embedding"):
        g = g.set_index("mode")
        bits = "  ".join(f"{m}={g.loc[m,'success']:.0f}%" for m in
                         ("stereo_flip", "ez_flip", "kekule") if m in g.index)
        print(f"     {emb:<22}{bits}")


if __name__ == "__main__":
    main()

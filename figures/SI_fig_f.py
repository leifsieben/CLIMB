"""SI Fig f — is the representation invariant to how a molecule is WRITTEN?

ONE script, ONE figure: figures_v2/SI_fig_f.png / .pdf   (panels a–c)

This is CLASS B: the two SMILES denote the SAME molecule, so a usable representation must NOT
separate them. The ten CLASS A modes -- genuinely different molecules, where the representation
must respond -- are fig_G in the main text. Both figures read this file's compute() and _panel(),
so the main-text and SI versions cannot disagree about a number.

THE TWO CLASSES ARE MEASURED ON DIFFERENT INPUT, AND THAT IS DELIBERATE
----------------------------------------------------------------------
  class A   RDKit-CANONICAL SMILES.  A chemistry question. Charging a sequence model for notation
            it never sees at inference answers the wrong question, so all strings are canonicalised.
  class B   the strings the pipeline ACTUALLY FEEDS ("as written"). Class B *is* the notation
            question, so canonicalising first would define it away.

The `input` column of the source CSV carries this per row and compute() asserts the mapping, so a
panel cannot silently be drawn on the wrong input -- the failure that produced three retracted
readings of this figure.

CANONICALISING FIRST DRIVES EVERY CLASS B CELL TO EXACTLY 0.000, FOR EVERY ARM. That is the
finding, not a missing result, and it is why this figure is an SI stress test rather than a
headline: the sensitivity shown here is a PIPELINE PROPERTY THAT IS ALREADY FIXED by embedding from
canonical SMILES, not a defect the models must be lived with. Panel (b), Kekulé, is the extreme
case -- and no molecule in our data is even written that way: 600/600 BACE SMILES use aromatic
notation. It is a stress test of the encoder, not an operational failure.

WHAT THE FIGURE SAYS
--------------------
1. THE FINGERPRINTS AND CheMeleon ARE INVARIANT BY CONSTRUCTION -- exactly 0.000 on all three
   panels. They read a graph, so the string cannot reach them.
2. THE CLMs ARE NOT. Re-written SMILES move CLIMB sup 0.373 and Kekulé form moves it 1.339 -- i.e.
   FURTHER than swapping in a completely different compound. Same molecule, same graph, different
   string, different embedding.
3. SYMMETRY-EQUIVALENT POSITIONS (c) ARE 0.000 FOR EVERYTHING, CLMs INCLUDED. This is the control
   that keeps (2) honest: the CLMs are not merely sensitive to every character-level edit. Where
   the canonicaliser resolves the ambiguity, they are invariant too.

RETRACTED, AND KEPT HERE SO IT IS NOT RE-DERIVED: "the CLMs cannot see stereochemistry" (they can,
weakly -- that is a magnitude statement and belongs to fig_G's bars, never a rate); "an untrained
encoder beats a pretrained one at stereo" (its own noise floor is larger than the effect); and any
framing of Kekulé as an operational failure (see above -- it never occurs in our data).

Data: figure_data/embedding_resolution/relative_response_figure.csv (peer session, commit 2c0bf7e).
Pairs, per-pair distances, the raw vectors and MCS-aligned depictions sit beside it.

Run:  python3 -m figures.SI_fig_f
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# The y axis is "response relative to a different molecule", the SAME unit in both classes -- but a
# LOW bar is the good outcome in class B and a HIGH bar is the good outcome in class A. The
# direction lives in the block label and the panel titles, so the axis stays neutral.

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS, SHADES

check_font()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "embedding_resolution" / "relative_response_figure.csv"
INK = "#000000"
TINT = "#F0EDE6"          # class B panel background; warm, so it reads as "different rule"

# (label in the CSV's `short` column, colour, legend label). EIGHT arms, the order Leif specified,
# grouped bare-then-descriptors so panel-to-panel the descriptor effect reads off adjacent pairs.
#
# Colours come from arms.py, never from a hex written here. The two CLIMB unsup entries are a
# MATCHED PAIR that exists only in this measurement -- they are not benchmark arms and are
# deliberately NOT registered in arms.py -- so they take two shades of the unsup family rather than
# the mainline `unsup` colour, which would falsely imply they are the arm the benchmark plots.
# Labels: "supervised" spelled out, "unsuperv." abbreviated, used the same way in every entry
# (user 2026-08-19). "ECFP4" rather than "ECFP4+stereo" because arms.py -- the single source of
# truth -- calls it ECFP4 everywhere else, and R3FP has chirality on too, so the suffix was
# marking a property both fingerprints share.
SERIES = [("ECFP",      ARMS["ecfp"]["color"],             "ECFP4"),
          ("ECFP+d",    ARMS["ecfp_desc"]["color"],        "ECFP4+desc"),
          ("r3fp",      ARMS["r3fp"]["color"],             "R3FP"),
          ("r3fp+d",    ARMS["r3fp_desc"]["color"],        "R3FP+desc"),
          ("uns-ENUM",  SHADES["unsup"][0],                "CLIMB unsuperv., augmented"),
          ("uns-CANON", SHADES["unsup"][2],                "CLIMB unsuperv., canonical"),
          ("sup",       ARMS["sup_dense"]["color"],        "CLIMB supervised"),
          ("CheMel",    ARMS["chemeleon_frozen"]["color"], "CheMeleon")]

# (class, mode, two-line panel title). The class blocks are drawn as separate figures.
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

# What each class is measured on. Asserted against the CSV in compute() rather than assumed: the
# whole figure inverts if a class is drawn on the other class's input.
INPUT_OF = {"A": "canonical", "B": "as_written"}


def compute():
    """{(klass, mode): relative response per arm}, one frame.

    `relative_response` is this model's response to the change divided by ITS OWN response to a
    genuinely different molecule of matched MW, so 1.00 means "moves the embedding as far as
    swapping in a different compound" and the reference is measured PER MODEL -- which is what lets
    a 512-d transformer and a 2048-bit fingerprint share one axis honestly. Effect size itself is
    the RMS per-dimension change in units of that dimension's spread over 1,000 background
    molecules. No threshold anywhere.
    """
    d = pd.read_csv(SRC)
    for kl, want in INPUT_OF.items():
        got = set(d.loc[d["klass"] == kl, "input"])
        assert got == {want}, (
            f"class {kl} must be measured on {want!r} input, found {sorted(got)}. Class A is a "
            f"chemistry question (canonical); class B IS the notation question (as written). "
            f"Drawing either on the other's input inverts the figure.")
    missing = {s for s, _, _ in SERIES} - set(d["short"])
    assert not missing, f"SI fig f: arms missing from {SRC.name}: {sorted(missing)}"
    return d.pivot_table(index=["klass", "mode"], columns="short", values="relative_response")


REF = 1.0             # the matched-MW reference: "as far as a different molecule"
YMAX = 1.78           # must clear matched_descriptors / R3FP = 1.617, the global max


def _panel(ax, vals, title, klass):
    """One mode. Bars are the response RELATIVE to swapping in a different molecule of matched MW.

    The reference line at 1.0 is the whole point of the unit: without it a reader has no way to
    know whether 0.68 is large. With it, fig_G panel (a) reads "an inverted stereocentre moves
    ECFP4 two thirds as far as a completely different compound, and CLIMB sup one hundredth as far".

    EXACT ZEROS ARE LABELLED. Half the cells in this figure are 0.000 by construction -- a
    fingerprint cannot see a re-written string, and Morgan invariants cannot see an isotope -- and
    an unlabelled flat baseline is indistinguishable from a bar that was never drawn. That
    ambiguity is the single most likely misreading here, so it is closed at the draw site.
    """
    if klass == "B":
        ax.set_facecolor(TINT)
    x = np.arange(len(SERIES))
    ax.bar(x, vals, width=0.80, color=[c for _, c, _ in SERIES],
           edgecolor=INK, linewidth=0.45, zorder=3)
    ax.axhline(REF, color=INK, ls=(0, (3, 2)), lw=0.7, zorder=4)
    for xi, v in zip(x, vals):
        if np.isfinite(v) and v == 0.0:
            ax.text(xi, YMAX * 0.022, "0", ha="center", va="bottom",
                    fontsize=FS["annot"] - 2.5, color=INK, zorder=5)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["0", "0.5", "1", "1.5"])
    ax.set_xticks([])
    ax.set_xlim(-0.70, len(SERIES) - 0.30)
    ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    ax.set_title(title, fontsize=FS["annot"], fontweight="bold", color=INK, pad=3, loc="center")


def _legend_handles():
    """Shared with fig_G so the two figures cannot drift in arm order, colour, or label."""
    h = [Patch(facecolor=c, edgecolor=INK, lw=0.6, label=lab) for _, c, lab in SERIES]
    h.append(Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.7,
                    label="= a different molecule (matched MW)"))
    return h


def report(R, modes, heading):
    """The printed table. Same resolution path as the bars, so the console cannot disagree."""
    print(f"\n{heading}\n")
    print(f"   {'mode':<22}" + "".join(f"{lab:>12s}" for lab, _, _ in SERIES))
    for kl, mode, _ in modes:
        row = f"   {mode:<22}"
        for lab, _, _ in SERIES:
            v = R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
            row += (f"{v:>12.3f}" if np.isfinite(v) else f"{'—':>12}")
        print(row)
    print("\n   1.000 = moves the embedding as far as a completely different compound of the same")
    print("   molecular weight. matched_mw is that reference and is 1.000 by construction.")


def main():
    R = compute()
    B = [m for m in MODES if m[0] == "B"]
    assert len(B) == 3, f"SI fig f expects the three class-B modes, found {len(B)}"

    fig = plt.figure(figsize=(STYLE["col2"], 2.60))
    gs = fig.add_gridspec(1, 3, left=0.135, right=0.995, top=0.815, bottom=0.375, wspace=0.34)
    for i, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gs[0, i])
        vals = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
                for lab, _, _ in SERIES]
        assert np.isfinite(vals).any(), f"SI fig f: no data for {mode}"
        _panel(ax, vals, title, kl)
        ax.text(0.0, 1.22, "abc"[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i == 0:
            ax.set_ylabel("response relative to a\ndifferent molecule", fontsize=FS["annot"])

    fig.text(0.030, (0.815 + 0.375) / 2, "Same molecule\nmust NOT separate", rotation=90,
             va="center", ha="center", fontsize=FS["annot"], fontweight="bold", color=INK)
    # The input note sits BELOW the legend, not above the panels: at this height a centred
    # header runs straight through panel (b)'s title. It is one line because it is the single
    # thing a reader must know to read the zeros correctly.
    fig.text(0.565, 0.020, "Input: SMILES as the pipeline writes them. Canonicalising first "
             "drives every cell in this figure to exactly 0.000.",
             ha="center", va="bottom", fontsize=FS["annot"] - 1.5, style="italic", color=INK)

    fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.565, 0.285),
               ncol=3, fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.5,
               labelspacing=0.35, borderpad=0.0, frameon=False)
    save(fig, "SI_fig_f")
    plt.close(fig)

    report(R, B, "SI Fig f (class B, as-written input) — same molecule, two spellings")


if __name__ == "__main__":
    main()

"""Fig G — does the representation respond when the CHEMISTRY changes?

ONE script, ONE figure: figures_v2/fig_G.png / .pdf   (panels a–j)

Ten chemical changes, 1,000 molecule pairs each. In every one the two molecules are GENUINELY
DIFFERENT, so a usable representation has to respond. "Respond" is measured in a unit the reader
can check on the same axis: the shift the change produces, divided by THAT MODEL'S OWN shift when
you swap in a completely different compound of matched molecular weight. 1.00 therefore means "this
change moves the embedding as far as changing the molecule". No threshold anywhere, and the
denominator is measured per model, so a 512-d transformer and a 2048-bit fingerprint sit on one
axis honestly. Panel (i) IS that reference and is 1.000 by construction — it stays in the grid so
the unit is visible rather than asserted.

INPUT IS RDKit-CANONICAL SMILES for the class-A panels and AS WRITTEN for the class-B controls
(k, l). Charging a sequence model for notation it never sees at inference would answer the wrong
question in class A; class B IS the notation question, and canonicalizing it would collapse both
controls to no-ops by construction. Nothing re-canonicalizes anything: the edits are applied to the
RDKit mol object and written out once. Under canonical input every representation here is
deterministic — re-embedding reproduces the vectors bit for bit, 128/128 — so there is no
significance question, only a size question, and this figure asks the size question.

THE UNIT WAS RE-EXAMINED ON 2026-08-25 and kept. A count of dimensions displaced past a threshold
was proposed and measured on this data: it tracks how DENSE a representation is rather than how
well it resolves an edit (ECFP4 moves 28 of 2048 bits for a completely different compound; CLIMB
418 of 512), and it is threshold-critical exactly where the CLM arms live. Full measurement in
notes/figG-resolution-metric.md.

WHAT THE FIGURE SAYS
--------------------
1,000 matched pairs per edit; sigma_j estimated on 10,000 unedited molecules that appear in no
pair. Whiskers are the INTERQUARTILE RANGE over the 1,000 pairs -- chemistry, not noise: every
representation here is deterministic, so there is nothing for an error bar to describe. Arm-vs-arm
claims below use the PAIRED contrast (resolution_contrasts.csv), where pair difficulty cancels.

1. THE STEREO GAP SURVIVES THE FAIR SETUP. An inverted stereocentre moves ECFP4 0.652-0.711 and
   the CLMs 0.009-0.033: twenty- to seventyfold, on canonical input, with no threshold. E/Z is
   0.879-0.959 against 0.013-0.063. The IQRs do not come close to touching
   (ECFP4 [0.47, 0.85] against CLIMB sup [0.007, 0.011]). This is the mechanism behind fig_C1's
   bare negative (unsupervised pretraining is worth -0.29% over a fine-tuned random init) rather
   than a restatement of it: the representation barely moves when the chemistry does.
2. DESCRIPTORS DILUTE STRUCTURAL SIGNAL. Adding the descriptor block makes ECFP4 worse on every
   class A mode: stereo 0.711 -> 0.652, C->N 0.765 -> 0.708, added methyl 0.762 -> 0.700, ring
   size 0.108 -> 0.103. Same story as the gap-narrowing result in fig_A2, arrived at
   independently. The one place it helps is isotopes (f): 0.000 -> 0.006, because the descriptor
   block carries exact mass while Morgan atom invariants do not.
3. RING SIZE IS THE COMMON BLIND SPOT (g). Cyclopentyl to cyclohexyl moves ECFP4 0.108 and the
   CLMs 0.043-0.072 -- the one class-A edit where no arm on this plate responds. Morgan r3-counts,
   computed but not drawn here, is the exception at 0.381; that is the strongest argument for a
   counts-based fingerprint in the whole measurement and it belongs in the SI rather than being
   silently dropped with the row.
4. AUGMENTATION DOES NOT BUY CHEMICAL SENSITIVITY -- and this is now a paired test, not a
   comparison of two point estimates. Against its matched canonical control the enumeration-
   augmented CLM is WORSE on eight of the ten class-A modes, every one of them at |t| > 7:
   added methyl -0.074 +/- 0.002, added fluorine -0.068 +/- 0.001, regioisomer -0.060 +/- 0.002,
   isotope -0.062 +/- 0.002, matched descriptors -0.105 +/- 0.003. It buys notation-invariance
   (re-written SMILES -0.122 +/- 0.002, panel k) and pays for it in chemistry. The reference mode
   itself comes out null (+0.004 +/- 0.004, t = 0.9), which is the internal check that the paired
   machinery is not manufacturing differences: both arms are divided by their own matched-MW
   median, so that one contrast MUST be zero.
5. THE BLIND SPOTS ARE COMPLEMENTARY. Isotopes (f) are the one place the CLMs win outright --
   0.205-0.279 against exactly 0.000 for bare ECFP4 -- because a [13C] token changes the string
   while Morgan atom invariants ignore it.
6. THE KEKULE CONTROL IS THE WORST RESULT ON THE PLATE AND MUST NOT BE SOFTENED. Panel (l) is the
   SAME molecule written aromatic or Kekule, so any response is an artefact. CLIMB supervised
   moves 1.343 [1.20, 1.46] -- FURTHER than swapping in a completely different compound of matched
   mass. The two unsupervised arms are 0.673-0.780. Both fingerprints are exactly 0.000. A reader
   who carries the class-A reading across gets this panel backwards, which is why the caption has
   to state the inversion explicitly.

A COMPARABILITY CAVEAT THAT BELONGS IN THE CAPTION. The two CLIMB unsup arms are a MATCHED PAIR
from climb_v2_h1 — both MLM, both 7,812 steps, differing only in augmentation — so point 4 is a
clean read. `CLIMB sup` is skip_dense_8M from phase 2 at 31,250 steps, FOUR TIMES the compute.
There is no supervised encoder at the h1 budget, so sup cannot be read against the other two as a
like-for-like comparison, only as "the mainline supervised arm".

SI FIG F IS GONE (user 2026-08-19: "I don't need SI f"). It was the class-B block — the same
molecule written two ways — and its two most informative modes are now panels (k) and (l) here.
The third, symmetry-equivalent positions, is 0.000 for every arm and lives in the source CSV; it
is worth one caption clause as the control that keeps (k) and (l) honest, since it shows the CLMs
are not merely sensitive to any character-level edit. This script now owns compute(), the panel
drawing and the legend outright, so there is no shared module left to drift.

Run:  python3 -m figures.fig_G
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX
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
# SHORTER, CONSISTENT CLIMB LABELS (Leif 2026-08-23: "CLIMB unsuperv., augmented e.g. is enough").
# Every CLIMB entry is now "CLIMB <objective>, <variant>" with no "frozen" -- it was on all three
# and distinguished none of them, and this figure has no fine-tuned arm for it to contrast with.
# CheMeleon dropped with the arm (figures.arms.RETIRED).
# R3FP DROPPED FROM THIS PLATE (Leif 2026-08-25): it carries no narrative anywhere else in the
# paper's main text. Its numbers are NOT deleted -- both variants are still computed and still
# written to relative_response_figure.csv, so the SI can quote them and the claim they supported
# (Morgan r3-counts is the most chemically responsive representation measured: stereo 0.823 vs
# ECFP4's 0.711, ring size 0.381 vs 0.108) is recoverable without re-running anything.
SERIES = [("ECFP",      ARMS["ecfp"]["color"],      "ECFP4"),
          ("ECFP+d",    ARMS["ecfp_desc"]["color"], "ECFP4+desc"),
          ("uns-ENUM",  SHADES["unsup"][0],         "CLIMB unsup., augmented"),
          ("uns-CANON", SHADES["unsup"][2],         "CLIMB unsup., canonical"),
          ("sup",       ARMS["sup_dense"]["color"], "CLIMB supervised")]

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

# What each class is measured on. Asserted against the CSV in compute() rather than assumed: the
# whole figure inverts if a class is drawn on the other class's input.
INPUT_OF = {"A": "canonical", "B": "as_written"}


def compute():
    """{(klass, mode): relative response per arm}, one frame.

    `relative_response` is this model's response to the change divided by ITS OWN response to a
    genuinely different molecule of matched MW, so 1.00 means "moves the embedding as far as
    swapping in a different compound" and the reference is measured PER MODEL -- which is what lets
    a 512-d transformer and a 2048-bit fingerprint share one axis honestly. Effect size itself is
    the RMS per-dimension change in units of that dimension's spread over 10,000 background
    molecules that appear in no reported pair. No threshold anywhere.
    """
    raw = pd.read_csv(SRC)
    # The source carries BOTH input conventions for every mode, so the selection has to be made
    # here -- and it has to be checked by counting what was selected, not by asking the frame what
    # it contains after the selection, which can only ever agree with itself.
    keep = []
    arms = [a for a, _, _ in SERIES]
    for kl, want in INPUT_OF.items():
        sel = raw[(raw["klass"] == kl) & (raw["input"] == want) & (raw["short"].isin(arms))]
        want_n = len(set(raw.loc[raw["klass"] == kl, "mode"])) * len(arms)
        assert len(sel) == want_n, (
            f"class {kl} on {want!r} input: {len(sel)} rows, expected {want_n} "
            f"({want_n // len(arms)} modes x {len(arms)} arms). Class A is a chemistry question "
            f"(canonical); class B IS the notation question (as written). Drawing either on the "
            f"other's input inverts the figure, and a missing row would silently drop a bar.")
        keep.append(sel)
    d = pd.concat(keep, ignore_index=True)
    missing = {s for s, _, _ in SERIES} - set(d["short"])
    assert not missing, f"fig_G: arms missing from {SRC.name}: {sorted(missing)}"
    n = sorted(set(d.loc[d["short"].isin([s for s, _, _ in SERIES]), "n"]))
    assert n and min(n) >= 1000, f"fig_G: expected >=1000 pairs per cell, found n in {n[:5]}"
    return {c: d.pivot_table(index=["klass", "mode"], columns="short", values=c)
            for c in ("relative_response", "q1", "q3")}


def _row(frame, kl, mode):
    """One panel's bar values, in SERIES order, NaN where the cell is absent."""
    return np.array([frame.loc[(kl, mode), lab] if (kl, mode) in frame.index else np.nan
                     for lab, _, _ in SERIES], dtype=float)


REF = 1.0             # the matched-MW reference: "as far as a different molecule"
# Must clear the largest DRAWN q3, not the largest median: with R3FP gone the tallest cell is
# ECFP4 on matched_descriptors, median 1.329 with q3 1.50, and CLIMB supervised on Kekule at
# q3 1.46. 1.78 was set for R3FP's 1.617 median and now wastes a fifth of every panel's height.
YMAX = 1.62


def _panel(ax, vals, lo, hi, title, klass):
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
    # WHISKERS ARE THE INTERQUARTILE RANGE OVER THE 1,000 PAIRS, not a standard error. They are
    # chemistry, not noise: an inverted stereocentre on a rigid ring is not the same edit as one on
    # a flexible chain, and every representation here is deterministic, so there is no measurement
    # noise for an error bar to describe. Arm-vs-arm claims use the PAIRED contrast
    # (figure_data/embedding_resolution/resolution_contrasts.csv) -- every arm sees the identical
    # pair list, so pair difficulty cancels there and these marginal ranges understate the
    # separation. Reading whisker overlap as "not different" is therefore wrong, and the caption
    # has to say so.
    err = np.vstack([np.clip(np.asarray(vals) - np.asarray(lo), 0, None),
                     np.clip(np.asarray(hi) - np.asarray(vals), 0, None)])
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.55,
                capsize=1.3, capthick=0.55, zorder=4)
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
    # The dashed reference line has NO legend entry (Leif 2026-08-23). The y-axis already reads
    # "response relative to a different molecule" and the line sits at 1.0 on it, so the entry
    # restated the axis and cost the legend a row.
    return [Patch(facecolor=c, edgecolor=INK, lw=0.6, label=lab) for _, c, lab in SERIES]


def report(R, modes, heading):
    """The printed table. Same resolution path as the bars, so the console cannot disagree."""
    print(f"\n{heading}\n")
    print(f"   {'mode':<22}" + "".join(f"{lab:>21s}" for lab, _, _ in SERIES))
    for kl, mode, _ in modes:
        row = f"   {mode:<22}"
        med, q1, q3 = (_row(R[c], kl, mode) for c in ("relative_response", "q1", "q3"))
        for k in range(len(SERIES)):
            row += (f"{med[k]:>7.3f}[{q1[k]:.2f},{q3[k]:.2f}]"
                    if np.isfinite(med[k]) else f"{'—':>21}")
        print(row)
    print("\n   median [IQR] over 1,000 pairs. 1.000 = moves the embedding as far as a completely")
    print("   different compound of the same molecular weight; matched_mw IS that reference, so")
    print("   its median is 1.000 by construction and its IQR is the reference's own spread.")




NCOL_A = 5          # the ten class-A modes, 2 rows x 5
NCOL = NCOL_A + 1   # plus a sixth column carrying the two class-B controls

# The two class-B panels shown here as CONTROLS (user 2026-08-19). They answer the opposite
# question -- same molecule written two ways, so a HIGH bar is a failure, not a response -- which
# is why they sit in their own column with the tinted background _panel() already gives class B,
# under their own header, rather than being mixed into the class-A grid. The full class-B block
# with the third mode (symmetry-equivalent positions) remains SI fig f.
CONTROLS = ["smiles_enumeration", "kekule"]


def main():
    R = compute()
    A = [m for m in MODES if m[0] == "A"]
    assert len(A) == 10, f"fig_G expects the ten class-A modes, found {len(A)}"
    B = [m for m in MODES if m[0] == "B" and m[1] in CONTROLS]
    B.sort(key=lambda m: CONTROLS.index(m[1]))
    assert len(B) == 2, f"fig_G expects the two class-B controls, found {[m[1] for m in B]}"

    fig = plt.figure(figsize=(STYLE["col2"], 3.85))
    gs = fig.add_gridspec(2, NCOL, left=0.068, right=0.995, top=0.875, bottom=0.250,
                          wspace=0.38, hspace=0.72)
    tags = "abcdefghij"
    for i, (kl, mode, title) in enumerate(A):
        ax = fig.add_subplot(gs[i // NCOL_A, i % NCOL_A])
        rates, q1, q3 = (_row(R[c], kl, mode) for c in ("relative_response", "q1", "q3"))
        assert np.isfinite(rates).any(), f"fig_G: no data for {mode}"
        _panel(ax, rates, q1, q3, title, kl)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL_A == 0:
            ax.set_ylabel("response relative to a\ndifferent molecule", fontsize=FS["annot"])

    # the control column
    for j, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gs[j, NCOL_A])
        vals, q1, q3 = (_row(R[c], kl, mode) for c in ("relative_response", "q1", "q3"))
        assert np.isfinite(vals).any(), f"fig_G: no data for control {mode}"
        _panel(ax, vals, q1, q3, title, kl)
        ax.text(0.0, 1.30, "kl"[j], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)

    # NO in-figure label on the control column (user 2026-08-19: it goes in the caption). The
    # inversion -- in (k) and (l) a HIGH bar is a failure -- is therefore carried by the tinted
    # background alone, and the caption MUST state it: a reader who carries the class-A reading
    # across gets both panels backwards. Keep that sentence in the caption if this figure is
    # ever re-cut.

    # ONE ROW (Leif 2026-08-25). Five entries fit across the plate once "unsuperv." is shortened
    # to "unsup."; the row does NOT widen the plate, which is still set by the axes at 6.74in.
    # Dropping the second legend row and closing the band it left took the plate 3.40in -> 3.17in.
    #
    # Historical note, kept because it is the constraint that decides this: the legend -- not the
    # axes -- sets this plate's width: savefig("tight")
    # grows the canvas to whatever hangs off it, so shrinking figsize does nothing here (tried:
    # 0.985 x col2 rendered to exactly the same 2040px). Spelling "frozen" into four labels
    # (user 2026-08-20) took the 5-column form to 6.80in against a 6.69in text block, and an
    # over-wide plate is downscaled by LaTeX, shrinking its fonts relative to every other figure.
    # 4 columns x 3 rows is narrower than 5 x 2 and costs one line of height, which this figure
    # has to spare. Verified by measuring the rendered PNG, not by eye.
    _h = _legend_handles()
    fig.legend(handles=_h, loc="upper center", bbox_to_anchor=(0.500, 0.212),
               # 9 handles, and one row overran badly: 6.73in -> 10.84in, far past the 6.69in
               # text block, so the plate would be scaled DOWN in LaTeX and every font with it.
               # Measured, not guessed. rows=3 restores the previous 3x3 block.
               ncol=row_ncol(_h, rows=1), fontsize=FS["legend"], handletextpad=0.4, columnspacing=1.0,
               labelspacing=0.35, borderpad=0.30, **LEGEND_BOX)
    save(fig, "fig_G")
    plt.close(fig)

    report(R, A + B, "Fig G — class A (canonical) plus the two class-B controls (as written)")


if __name__ == "__main__":
    main()

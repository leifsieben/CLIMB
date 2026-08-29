"""Fig G — can a model TELL THE PAIR APART? Resolution as held-out classification AUC.

ONE script, ONE figure: figures_v2/fig_G.png / .pdf   (panels a–k)

Eleven chemical changes, 1,000 molecule pairs each. For one (arm, edit): label every parent 0 and
its edited partner 1, fit a gradient-boosted tree, report ROC-AUC on held-out molecules. 0.5 = the
edit leaves no signature a model can learn. 1.0 = every held-out pair separated. No threshold, no
normalisation, no free parameter, and scale-free, so a 2048-bit fingerprint and a 512-d
transformer sit on one axis with nothing to calibrate.

THE AXIS RUNS 0.5 TO 1.0 AND THAT IS THE POINT (Leif 2026-08-28): "a zero bar must always mean
not resolved -- if you can't see a bar you know immediately a prediction also can't see this
change." An empty bar is therefore a claim, and the plate only earns it because every panel where
an empty bar could have meant something ELSE was removed rather than captioned. See MODES.

WHY NOT THE MAGNITUDE RATIO THIS FIGURE USED UNTIL 2026-08-28
------------------------------------------------------------
It reported the RMS per-dimension shift divided by the arm's own shift when a different compound
of matched MW was substituted. Leif's objection: XGBoost splits on SINGLE dimensions, so a model
does not need a large shift, it needs ONE coordinate that separates the pair -- and a magnitude
axis systematically understates a representation whose information is concentrated. One flipped
bit in 2048 is a negligible norm change however decisive it is.

A dimension COUNT was tried first and rejected on measurement (notes/figG-resolution-metric.md):
threshold-critical exactly where the CLM arms live. Classification inherits neither problem, and
it changed conclusions rather than restating them -- the HUME session measured CheMeleon at 0.544
on stereo, barely above chance, where the ratio axis had called it strongly stereo-sensitive. The
magnitude was noise pointing in inconsistent directions.

THE OLD METRIC CAN NO LONGER BE RECOMPUTED. Its denominator was the matched_mw mode, which is gone
from the pair set (Leif: "100% not needed"), and scripts/resolution_relative_response.py asserts on
finding those pairs. This is a one-way door, taken deliberately.

HOW TO READ IT
--------------
* SPLIT BY PAIR, NEVER BY MOLECULE, and by connected COMPONENT of pairs sharing a molecule --
  molecules recur across pairs, so a pair-level split leaks and the leak RAISES the score, which
  is the direction that flatters. The harness asserts the halves share no molecule.
* THE DASHED LINE IS THE FREE-INFORMATION FLOOR: character 1-2-grams of the SMILES, no chemistry,
  same harness. A bar below it carries LESS about the edit than counting characters does. It is a
  bound on what a CLM's score PROVES, not a mechanism claim about the fingerprints -- ECFP4
  resolves stereochemistry without seeing a character, because chirality is in the atom invariant.
* A "0" MARKS A COLLISION-DOMINATED CELL: the arm maps both members of most pairs to the SAME vector,
  so it cannot possibly differ whatever head is put on top. "Not resolved" in its strongest form,
  needing no threshold and no metric.
* PANELS (j) AND (k) INVERT -- same molecule written two ways, so a HIGH bar is a failure. Carried
  by the tinted background; the caption must say it.
* Whiskers are +/- 1 SD over 5 SPLIT SEEDS. The representations are deterministic and the arms
  have no training seed, so the split is the only stochastic element: this is the spread of the
  ESTIMATE, not of the model. Median 0.009.

WHAT THE FIGURE SAYS
--------------------
1. THE STEREO GAP SURVIVES THE METRIC CHANGE AND READS PLAINLY (a). ECFP4 0.764, CLIMB supervised
   0.495 -- chance. Not "moves 20-70x less", which needed a calibration sentence: the fingerprint
   separates three quarters of held-out pairs and the CLM separates none. This is the mechanism
   behind fig_C1's bare negative rather than a restatement of it.
2. THE CLM ARMS SIT BELOW THE FREE-INFORMATION FLOOR ON SIX OF NINE CLASS-A EDITS. On (b) the
   supervised CLM is 0.408 BELOW character n-grams of the strings it was trained on. Only ring
   size (h) and para-vs-meta (i) show every trained arm clearing the floor -- the two edits the
   string genuinely cannot hand over. That is a narrower and much stronger claim than "the
   representation barely moves", and it is only visible because the floor is drawn.
3. ISOTOPES ARE TWO CLAIMS, NOT ONE (g). "Morgan atom invariants cannot represent a same-element
   isotope substitution" survives and is airtight: ECFP4 collides on 952 of 1,000 pairs. "The CLM
   sees isotopes" does NOT survive as stated -- the floor is 0.999, so the [13C] TOKEN carries it
   and no chemistry is required. The real result is ECFP4+desc at 0.783 with zero collisions: the
   descriptor block carries exact mass.
4. HOMOLOGATION SEPARATES THE ARMS WHERE ADDING A METHYL DOES NOT (e vs d). A methyl BRANCHES the
   skeleton and is visible in the string (floor 0.925); a CH2 insertion extends it and leaves
   every functional group intact (floor 0.693). ECFP4 clears the floor by +0.157 there against
   +0.057 on the methyl, and the supervised CLM clears it by exactly 0.000.
5. PARA VS META, NOT ORTHO VS META (i). The old pairing read 1.000 for all seven arms INCLUDING an
   untrained random encoder, because ortho is written with its second substituent TERMINAL and the
   other two carry a branch -- the panel measured our template. Leif's rule: an edit must only ever
   MOVE a branch, never CREATE one. Rebuilt, the floor drops to 0.522 and an untrained encoder to
   0.534 while every trained arm holds 0.967-1.000. It is now the cleanest evidence on the plate
   that pretraining encodes something the string does not hand over.

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
SRC = ROOT / "figure_data" / "embedding_resolution" / "separability_auc.csv"
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
# NINE class-A edits and TWO class-B controls. Three panels were REMOVED on 2026-08-28 under a
# rule of Leif's that the metric change forced: AN EMPTY BAR MUST MEAN EXACTLY ONE THING, "not
# resolved". A panel where no arm can clear chance cannot distinguish "nobody resolves this" from
# "the question has no answer", and captioning that ambiguity is not fixing it.
#   matched_mw / matched_descriptors  -- two UNRELATED molecules, so which one is "A" is
#       arbitrary and nothing generalises to a held-out pair. Every arm reads ~0.5 however well it
#       sees the difference: ECFP4 puts matched_mw pairs at Tanimoto 0.099, ~29 bits apart, zero
#       identical vectors, and still scores 0.509. That is an ill-posed question, not a blind
#       representation. matched_mw is gone from the pair set entirely (Leif: "100% not needed").
#   symmetry_equivalent -- all 1,000 pairs are the SAME STRING, so both classes are one vector for
#       every arm and 0.500 is forced. Vacuous under either metric. It remains in SI fig f.
# The mirror of that rule, from the HUME session: a panel where EVERY arm hits 1.000 measures the
# pair generator too. That was regioisomer, and it is why it is now para-vs-meta (below).
MODES = [("A", "stereo_flip",         "Inverted\nstereocentre"),
         ("A", "ez_flip",             "Flipped E/Z\ndouble bond"),
         ("A", "c_to_n",              "Aromatic C→N\n(benzene→pyridine)"),
         ("A", "add_methyl",          "One methyl\nadded"),
         ("A", "ch2_homologue",       "One CH$_2$ inserted\n(homologue)"),
         ("A", "add_fluorine",        "One fluorine\nadded"),
         ("A", "isotope_13c",         "$^{12}$C→$^{13}$C,\ngraph unchanged"),
         ("A", "ring_size",           "Cyclopentyl ↔\ncyclohexyl"),
         ("A", "regioisomer",         "para vs meta\nsubstitution"),
         ("B", "smiles_enumeration",  "Re-written\nSMILES"),
         ("B", "kekule",              "Kekulé\nform"),
         ("B", "symmetry_equivalent", "Equivalent\npositions")]

# The free-information floor, drawn as a line in every panel. Character 1-2-grams of the SMILES,
# no chemistry, through the identical harness -- so whatever it reaches is available to any string
# model for nothing. It is a bound on what a CLM's score PROVES, not a mechanism claim about the
# fingerprints: ECFP4 resolves stereochemistry without seeing a character, because chirality is in
# the atom invariant. Idea from the HUME figures session, 2026-08-28.
NOTATION = "notation"

# What each class is measured on. Asserted against the CSV in compute() rather than assumed: the
# whole figure inverts if a class is drawn on the other class's input.
INPUT_OF = {"A": "canonical", "B": "as_written"}


def compute():
    """{(klass, mode): held-out AUC per arm}, plus the notation floor and the collision counts.

    The value is the ROC-AUC of a gradient-boosted tree asked to tell an edited molecule from its
    parent, on pairs it has never seen -- 0.5 is chance, 1.0 separates every held-out pair. Split
    by CONNECTED COMPONENT of molecules, so a molecule appearing in two pairs cannot sit on both
    sides; the harness asserts it rather than intending it.
    """
    raw = pd.read_csv(SRC)
    arms = [a for a, _, _ in SERIES]
    keep = []
    for kl, want in INPUT_OF.items():
        sel = raw[(raw["klass"] == kl) & (raw["input"] == want)
                  & (raw["short"].isin(arms + [NOTATION]))]
        want_n = len(set(raw.loc[raw["klass"] == kl, "mode"])) * (len(arms) + 1)
        assert len(sel) == want_n, (
            f"class {kl} on {want!r} input: {len(sel)} rows, expected {want_n}. Class A is a "
            f"chemistry question (canonical); class B IS the notation question (as written). "
            f"Drawing either on the other's input inverts the figure.")
        keep.append(sel)
    d = pd.concat(keep, ignore_index=True)
    missing = set(arms + [NOTATION]) - set(d["short"])
    assert not missing, f"fig_G: arms missing from {SRC.name}: {sorted(missing)}"
    n = sorted(set(d["n_pairs"]))
    assert n and min(n) >= 900, f"fig_G: expected ~1000 pairs per cell, found n in {n[:5]}"
    return {c: d.pivot_table(index=["klass", "mode"], columns="short", values=c)
            for c in ("auc_mean", "auc_sd", "n_degenerate", "n_pairs")}


def _row(frame, kl, mode):
    """One panel's bar values, in SERIES order, NaN where the cell is absent."""
    return np.array([frame.loc[(kl, mode), lab] if (kl, mode) in frame.index else np.nan
                     for lab, _, _ in SERIES], dtype=float)


CHANCE = 0.5          # an unresolved edit. The axis FLOOR, so an unresolved cell has no bar.
# HEADROOM ABOVE 1.0, or a perfectly-separating arm loses its whisker. Four arms read exactly
# 1.000 on para-vs-meta; with the axis ending at 1.0 their bar tops sit ON the boundary and the
# error-bar caps are clipped away, so the panel looked like the only one with no uncertainty
# drawn. The SD really is 0.000 there (every split seed separates every held-out pair), and that
# is worth SEEING as a drawn cap rather than inferring from an absence. Ticks stay at 0.5 and 1.0,
# so the axis still reads as "chance to perfect".
YMAX = 1.035

# Sub-chance cells. AUC scatters slightly below 0.5 on an unresolved edit, and on a [0.5, 1] axis
# those clip to nothing -- indistinguishable from a panel with no data. A tick at the baseline
# says "measured, at chance" without spending height or a legend key.
SUBTICK = 0.006


def _panel(ax, vals, sds, degen, npair, floor, title, klass):
    """One edit. Bar height is how well a tree separates the pair on held-out molecules.

    THE AXIS STARTS AT CHANCE (Leif 2026-08-28): "a zero bar must always mean not resolved --
    if you can't see a bar you know immediately a prediction also can't see this change." That is
    only honest once every panel where an empty bar could mean something else has been removed
    from the plate, which is why three modes are gone (see MODES).

    A COLLISION IS THE SAME STATEMENT, HARDER. Where an arm maps both members of a pair to the
    same vector it cannot possibly differ, whatever head is put on top -- no threshold, no metric.
    Cells that are mostly collisions are marked, so the strongest form of "not resolved" is
    legible rather than being one more short bar: ECFP4 collides on 952 of 1,000 isotope pairs.

    THE DOTTED LINE IS THE FREE-INFORMATION FLOOR -- character n-grams of the SMILES, no
    chemistry. A bar below it carries LESS about the edit than counting characters does.
    """
    if klass == "B":
        ax.set_facecolor(TINT)
    x = np.arange(len(SERIES))
    v = np.asarray(vals, dtype=float)
    drawn = np.clip(v, CHANCE, None)
    ax.bar(x, drawn - CHANCE, bottom=CHANCE, width=0.80,
           color=[c for _, c, _ in SERIES], edgecolor=INK, linewidth=0.45, zorder=3)
    # Measured-at-chance, so an empty slot cannot be read as a missing arm.
    for xi, val in zip(x, v):
        if np.isfinite(val) and val <= CHANCE:
            ax.bar([xi], [SUBTICK], bottom=CHANCE, width=0.80, color=INK, lw=0, zorder=4)
    # +/- 1 SD over the FIVE SPLIT SEEDS. Every representation here is deterministic and the arms
    # have no training seed, so the split is the only stochastic element and this is the spread of
    # the ESTIMATE, not of the model. Median SD is 0.009, so these are small by construction --
    # they are drawn because a whisker that is invisible because it is tight looks the same as one
    # that was never computed, and only one of those is a claim.
    err = np.asarray(sds, dtype=float)
    # Includes cells whose SD is exactly 0.0 -- that is a measured result (five seeds, identical
    # answer), not a missing one, and an undrawn cap is indistinguishable from an uncomputed one.
    ok = np.isfinite(v) & np.isfinite(err) & (v > CHANCE)
    if ok.any():
        ax.errorbar(x[ok], v[ok], yerr=err[ok], fmt="none", ecolor=INK, elinewidth=0.55,
                    capsize=1.3, capthick=0.55, zorder=5)
    if np.isfinite(floor):
        ax.axhline(floor, color=INK, ls=(0, (2, 1.6)), lw=0.8, zorder=6)
    for xi, dg, npr in zip(x, degen, npair):
        if np.isfinite(dg) and np.isfinite(npr) and npr and dg / npr >= 0.5:
            ax.text(xi, CHANCE + 0.012, "0", ha="center", va="bottom",
                    fontsize=FS["annot"] - 1.5, color=INK, zorder=7)
    ax.set_ylim(CHANCE, YMAX)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.5", "", "1"])
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
    h = [Patch(facecolor=c, edgecolor=INK, lw=0.6, label=lab) for _, c, lab in SERIES]
    # The floor DOES get a key, unlike the old 1.0 reference line: that line restated the axis,
    # this one introduces a quantity the axis says nothing about.
    h.append(Line2D([], [], color=INK, ls=(0, (2, 1.6)), lw=0.9,
                    label="notation floor (char n-grams)"))
    return h


def report(R, modes, heading):
    """The printed table. Same resolution path as the bars, so the console cannot disagree."""
    print(f"\n{heading}\n")
    print(f"   {'mode':<22}{'floor':>8}" + "".join(f"{lab:>15s}" for lab, _, _ in SERIES))
    for kl, mode, _ in modes:
        row = f"   {mode:<22}"
        auc, sd, dg, npr = (_row(R[c], kl, mode) for c in
                            ("auc_mean", "auc_sd", "n_degenerate", "n_pairs"))
        fl = R["auc_mean"].loc[(kl, mode), NOTATION] if (kl, mode) in R["auc_mean"].index else np.nan
        row += f"{fl:>8.3f}" if np.isfinite(fl) else f"{'—':>8}"
        for k in range(len(SERIES)):
            if not np.isfinite(auc[k]):
                row += f"{'—':>15}"
                continue
            mark = "0" if npr[k] and dg[k] / npr[k] >= 0.5 else " "
            row += f"{auc[k]:>8.3f}±{sd[k]:.3f} {mark}"
        print(row)
    print("\n   held-out ROC-AUC, mean ± 1 SD over 5 split seeds. 0.500 = the edit leaves no")
    print("   signature a tree can learn; 1.000 = every held-out pair separated.")
    print("   'floor' = character n-grams of the SMILES, no chemistry: available to any string")
    print("   model for free, so a bar below it carries less than the raw text does.")
    print("   0 marks a cell where the arm maps BOTH members of most pairs to the same vector —")
    print("   it cannot possibly differ, which is 'not resolved' in its strongest form.")


NCOL_A = 5          # nine class-A modes across 2 rows of 5; the tenth slot is left empty
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
    assert len(A) == 9, f"fig_G expects the nine class-A modes, found {[m[1] for m in A]}"
    B = [m for m in MODES if m[0] == "B" and m[1] in CONTROLS]
    B.sort(key=lambda m: CONTROLS.index(m[1]))
    assert len(B) == 2, f"fig_G expects the two class-B controls, found {[m[1] for m in B]}"

    fig = plt.figure(figsize=(STYLE["col2"], 3.85))
    gs = fig.add_gridspec(2, NCOL, left=0.068, right=0.995, top=0.875, bottom=0.250,
                          wspace=0.38, hspace=0.72)
    tags = "abcdefghi"
    for i, (kl, mode, title) in enumerate(A):
        ax = fig.add_subplot(gs[i // NCOL_A, i % NCOL_A])
        auc, sd, dg, npr = (_row(R[c], kl, mode) for c in
                            ("auc_mean", "auc_sd", "n_degenerate", "n_pairs"))
        floor = R["auc_mean"].loc[(kl, mode), NOTATION]
        assert np.isfinite(auc).any(), f"fig_G: no data for {mode}"
        _panel(ax, auc, sd, dg, npr, floor, title, kl)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL_A == 0:
            ax.set_ylabel("can a tree tell the pair\napart?  (held-out AUC)",
                          fontsize=FS["annot"])

    # the control column
    for j, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gs[j, NCOL_A])
        auc, sd, dg, npr = (_row(R[c], kl, mode) for c in
                            ("auc_mean", "auc_sd", "n_degenerate", "n_pairs"))
        floor = R["auc_mean"].loc[(kl, mode), NOTATION]
        assert np.isfinite(auc).any(), f"fig_G: no data for control {mode}"
        _panel(ax, auc, sd, dg, npr, floor, title, kl)
        ax.text(0.0, 1.30, "jk"[j], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)

    # NO in-figure label on the control column (user 2026-08-19: it goes in the caption). The
    # inversion -- in (j) and (k) a HIGH bar is a failure -- is therefore carried by the tinted
    # background alone, and the caption MUST state it: a reader who carries the class-A reading
    # across gets both panels backwards. Keep that sentence in the caption if this figure is
    # ever re-cut.
    _h = _legend_handles()
    fig.legend(handles=_h, loc="upper center", bbox_to_anchor=(0.500, 0.212),
               ncol=row_ncol(_h, rows=1), fontsize=FS["legend"], handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.35, borderpad=0.30, **LEGEND_BOX)
    save(fig, "fig_G")
    plt.close(fig)

    report(R, A + B, "Fig G — class A (canonical) plus the two class-B controls (as written)")


if __name__ == "__main__":
    main()

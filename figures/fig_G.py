"""Fig G — does the representation respond when the CHEMISTRY changes?

ONE script, ONE figure: figures_v2/fig_G.png / .pdf   (panels a–j)

Ten chemical changes, 100 molecule pairs each. In every one the two molecules are GENUINELY
DIFFERENT, so a usable representation has to respond. "Respond" is measured in a unit the reader
can check on the same axis: the shift the change produces, divided by THAT MODEL'S OWN shift when
you swap in a completely different compound of matched molecular weight. 1.00 therefore means "this
change moves the embedding as far as changing the molecule". No threshold anywhere, and the
denominator is measured per model, so a 512-d transformer and a 2048-bit fingerprint sit on one
axis honestly. Panel (i) IS that reference and is 1.000 by construction — it stays in the grid so
the unit is visible rather than asserted.

INPUT IS RDKit-CANONICAL SMILES for every arm. Charging a sequence model for notation it never sees
at inference would answer the wrong question; the notation question is class B, SI fig f. Under
canonical input every representation here is deterministic — re-embedding reproduces the vectors
bit for bit, 128/128 — so there is no significance question, only a size question, and this figure
asks the size question.

WHAT THE FIGURE SAYS
--------------------
1. THE STEREO GAP SURVIVES THE FAIR SETUP. An inverted stereocentre moves the fingerprints
   0.64–0.92 and the CLMs 0.006–0.030: twenty- to hundredfold, on canonical input, with no
   threshold. E/Z is 0.95–1.06 against 0.013–0.059. Across the ten changes the CLMs sit at
   0.006–0.49 where the fingerprints sit at 0.000–1.62. This is the mechanism behind fig_C1's bare
   negative (unsupervised pretraining is worth −0.29% over a fine-tuned random init) rather than a
   restatement of it: the representation barely moves when the chemistry does.
2. MORGAN r3-COUNTS IS THE MOST CHEMICALLY RESPONSIVE REPRESENTATION TESTED, and it is not close
   where ECFP4 is weakest: stereo 0.920 vs 0.679, ring size 0.388 vs 0.104. That is independent
   support for the third XGBoost anchor from a measurement with no benchmark score in it.
3. DESCRIPTORS DILUTE STRUCTURAL SIGNAL. Adding the descriptor block makes BOTH fingerprints
   slightly worse on every class A mode (r3-counts stereo 0.920 → 0.856, ring size 0.388 → 0.362).
   Same story as the gap-narrowing result in fig_A2, arrived at independently.
4. AUGMENTATION DOES NOT BUY CHEMICAL SENSITIVITY. Against its matched canonical control, the
   enumeration-augmented CLM is no better on stereo (0.027 vs 0.030) and WORSE on added methyl
   (0.140 vs 0.234), added fluorine, ring size and matched descriptors. It buys notation-invariance
   (SI fig f a: 0.243 vs 0.376) and pays for it in chemistry. A genuine negative, and only
   measurable because that control exists.
5. THE BLIND SPOTS ARE COMPLEMENTARY. Isotopes (f) are the one place the CLMs win outright —
   0.18–0.27 against exactly 0.000 for both bare fingerprints and CheMeleon, because a [13C] token
   changes the string while Morgan atom invariants ignore it. Ring size (g) is weak for everything
   except the r3-counts pair.

A COMPARABILITY CAVEAT THAT BELONGS IN THE CAPTION. The two CLIMB unsup arms are a MATCHED PAIR
from climb_v2_h1 — both MLM, both 7,812 steps, differing only in augmentation — so point 4 is a
clean read. `CLIMB sup` is skip_dense_8M from phase 2 at 31,250 steps, FOUR TIMES the compute.
There is no supervised encoder at the h1 budget, so sup cannot be read against the other two as a
like-for-like comparison, only as "the mainline supervised arm".

Class B — the same molecule written two ways — is SI fig f. Both figures read the same table
through the same code (figures/SI_fig_f.py owns compute(), the panel drawing and the legend), so
the main-text and SI versions can never disagree about a number.

Run:  python3 -m figures.fig_G
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, check_font
from figures.SI_fig_f import compute, _panel, report, _legend_handles, MODES, SERIES

check_font()
INK = "#000000"

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
        rates = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
                 for lab, _, _ in SERIES]
        assert np.isfinite(rates).any(), f"fig_G: no data for {mode}"
        _panel(ax, rates, title, kl)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL_A == 0:
            ax.set_ylabel("response relative to a\ndifferent molecule", fontsize=FS["annot"])

    # the control column
    for j, (kl, mode, title) in enumerate(B):
        ax = fig.add_subplot(gs[j, NCOL_A])
        vals = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
                for lab, _, _ in SERIES]
        assert np.isfinite(vals).any(), f"fig_G: no data for control {mode}"
        _panel(ax, vals, title, kl)
        ax.text(0.0, 1.30, "kl"[j], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)

    # NO in-figure label on the control column (user 2026-08-19: it goes in the caption). The
    # inversion -- in (k) and (l) a HIGH bar is a failure -- is therefore carried by the tinted
    # background alone, and the caption MUST state it: a reader who carries the class-A reading
    # across gets both panels backwards. Keep that sentence in the caption if this figure is
    # ever re-cut.

    # 2 rows x 5 columns for the 9 entries (8 arms + the reference line), user request. Checked
    # against the canvas below rather than assumed -- a legend wider than the figure is what
    # pushes savefig's tight bbox out and silently rescales the whole plate in LaTeX.
    fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.500, 0.175),
               ncol=5, fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.2,
               labelspacing=0.35, borderpad=0.0, frameon=False)
    save(fig, "fig_G")
    plt.close(fig)

    report(R, A + B, "Fig G — class A (canonical) plus the two class-B controls (as written)")


if __name__ == "__main__":
    main()

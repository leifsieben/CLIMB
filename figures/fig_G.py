"""Fig G — can the representation tell two different molecules apart at all?

ONE script, ONE figure: figures_v2/fig_G.png / .pdf   (panels a–j)

Ten chemical changes, 100 molecule pairs each. In every one the two molecules are GENUINELY
DIFFERENT, so a usable representation has to respond. "Respond" is measured in a unit the reader
can check on the same axis: the shift the change produces, divided by THAT MODEL'S OWN shift when
you swap in a completely different compound of matched molecular weight. 1.00 therefore means "this
change moves the embedding as far as changing the molecule". No threshold anywhere, and the
denominator is measured per model, so a 512-d transformer and a 2048-bit fingerprint sit on one
axis honestly. Panel (i) IS that reference and is 1.00 by construction — it stays in the grid so
the unit is visible rather than asserted.

THE RESULT: an inverted stereocentre moves ECFP4 0.68 — two thirds as far as a different molecule —
and CLIMB sup 0.009. Seventy-fold. E/Z is 1.02 against 0.013. Across the ten changes the CLMs sit
at 0.01–0.35 where the fingerprints sit at 0.10–1.36. This is the mechanism behind fig_C1's bare
negative (unsupervised pretraining is worth −0.29% over a fine-tuned random init) rather than a
restatement of it: the representation barely moves when the chemistry does.

ring_size (g) is the one change every representation is weak on, 0.07–0.11 across the board, and
isotopes (f) are the one place the CLMs win outright — 0.21/0.29 against exactly 0.000 for ECFP4
and CheMeleon, because a [13C] token changes the string while Morgan atom invariants ignore it.
Complementary blind spots, in a threshold-free unit.

A NOTE ON WHAT THIS METRIC REPLACED, because three earlier versions of this figure reported
otherwise. A noise-calibrated variant thresholded each change against the model's response to a
re-written SMILES, which drove every CLM cell to 0%. That test was withdrawn: in the pipeline every
molecule is embedded from its CANONICAL SMILES, so re-writing never occurs at inference, and
charging the CLMs for it was measuring a property they are never asked to have. Under canonical
input every representation here is deterministic — re-embedding reproduces the vectors bit for bit,
128/128 — so there is no significance question, only a size question, and this figure asks the size
question.

Class B — the same molecule written two ways — is SI fig f, and it is the sharper half: re-writing
a molecule in Kekulé form moves CLIMB sup 1.34, i.e. FURTHER than swapping in a different compound.
Both figures read the same table through the same code (figures/SI_fig_f.py owns compute() and the
panel drawing), so the main-text and SI versions can never disagree about a number.

Run:  python3 -m figures.fig_G
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS
from figures.SI_fig_f import compute, _panel, MODES, SERIES, NCOL

check_font()
INK = "#000000"


def main():
    R = compute()
    A = [m for m in MODES if m[0] == "A"]
    assert len(A) == 10, f"fig_G expects the ten class-A modes, found {len(A)}"

    fig = plt.figure(figsize=(STYLE["col2"], 3.55))
    gs = fig.add_gridspec(2, NCOL, left=0.075, right=0.995, top=0.865, bottom=0.215,
                          wspace=0.34, hspace=0.72)
    tags = "abcdefghij"
    for i, (kl, mode, title) in enumerate(A):
        ax = fig.add_subplot(gs[i // NCOL, i % NCOL])
        rates = [R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
                 for lab, _, _ in SERIES]
        assert np.isfinite(rates).any(), f"fig_G: no data for {mode}"
        _panel(ax, rates, title, kl)
        ax.text(0.0, 1.30, tags[i], transform=ax.transAxes, fontsize=FS["panel_tag"],
                fontweight="bold", va="bottom", ha="left", color=INK)
        if i % NCOL == 0:
            ax.set_ylabel("response relative to a\ndifferent molecule", fontsize=FS["annot"])

    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=ARMS[k]["color"], edgecolor=INK, lw=0.7, label=lab)
               for _, k, lab in SERIES]
    handles.append(Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.7,
                          label="= a different molecule (matched MW)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.118), ncol=6,
               fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.4, borderpad=0.0,
               frameon=False)
    save(fig, "fig_G")
    plt.close(fig)

    print("\nFig G — response relative to swapping in a different molecule (matched MW)\n")
    print(f"   {'mode':<22}" + "".join(f"{lab:>14s}" for lab, _, _ in SERIES))
    for kl, mode, _ in A:
        row = f"   {mode:<22}"
        for lab, _, _ in SERIES:
            v = R.loc[(kl, mode), lab] if (kl, mode) in R.index else np.nan
            row += f"{v:>14.3f}" if np.isfinite(v) else f"{'—':>14}"
        print(row)
    print("\n   1.000 = moves the embedding as far as a completely different compound of the")
    print("   same molecular weight. matched_mw is that reference and is 1.000 by construction.")


if __name__ == "__main__":
    main()

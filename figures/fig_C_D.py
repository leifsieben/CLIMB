"""Fig C -- the assembled similarity & transfer figure: C1 + C2 + D in one canvas, panels a-f.

ONE script, ONE figure: figures_v2/fig_C_D.png / .pdf

Layout (user request 2026-08-17: assemble C1/C2/D with one consistent a-f scheme, space
efficiently): two rows, one per QUESTION --

  top row (molecular similarity: pretraining data <-> downstream molecules)
    a  C1 bars     Unsupervised: who benefits (identity vs similarity vs novelty)
    b  C1 trend    Unsupervised: lift vs corpus Tanimoto, per task
    c  C2 scatter  Supervised: family-task lift vs family-task Tanimoto (H10 test)
  bottom row (task similarity: which SFT label type transfers where)
    d  D bars      Which SFT label type helps on average
    e  D matrix    SFT family x eval task transfer (symlog)
    f  D slope     Dense vs sparse-all across task groups (descriptor-like mapping)

No analysis code lives here: fig_C1/fig_C2/fig_D expose compute() + draw() and this script only
arranges their axes, so the standalone figures and this assembly can never drift apart.

Run:  python3 -m figures.fig_C_D
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, title, check_font
from figures import fig_C1, fig_C2, fig_D

check_font()


def main():
    d1 = fig_C1.compute()
    d2 = fig_C2.compute()
    d3 = fig_D.compute()

    fig = plt.figure(figsize=(STYLE["col2"], 6.5))
    row1, row2 = fig.subfigures(2, 1, height_ratios=[1.0, 1.04], hspace=0.05)

    # both rows share ONE left-to-right width and the same column ratios, so the upper and lower
    # halves start/end at the same x (user 2026-08-17; possible since c's legend moved inside and
    # d's long parenthetical labels were dropped)
    # --- top row: molecular similarity ---------------------------------------------------------
    gs1 = row1.add_gridspec(1, 3, width_ratios=[1.05, 1.20, 1.25], wspace=0.38,
                            left=0.125, right=0.985, top=0.90, bottom=0.16)
    a1 = row1.add_subplot(gs1[0, 0])
    a2 = row1.add_subplot(gs1[0, 1])
    a3 = row1.add_subplot(gs1[0, 2])
    fig_C1.draw(a1, a2, d1, tags=("a", "b"), compact=True)
    fig_C2.draw(a3, d2, tag="c", compact=True)

    # --- bottom row: task similarity -----------------------------------------------------------
    gs2 = row2.add_gridspec(1, 3, width_ratios=[1.05, 1.20, 1.25], wspace=0.38,
                            left=0.125, right=0.985, top=0.90, bottom=0.19)
    b1 = row2.add_subplot(gs2[0, 0])
    b2 = row2.add_subplot(gs2[0, 1])
    b3 = row2.add_subplot(gs2[0, 2])
    fig_D.draw(b1, b2, b3, d3, tags=("d", "e", "f"), compact=True)

    # row group labels: rotated, at the very left edge of the figure (user 2026-08-17)
    fig.text(0.002, 0.755, "Molecular Similarity", rotation=90, va="center", ha="center",
             fontsize=FS["panel_tag"], fontweight="bold")
    fig.text(0.002, 0.260, "Task Similarity", rotation=90, va="center", ha="center",
             fontsize=FS["panel_tag"], fontweight="bold")

    save(fig, "fig_C_D")
    plt.close(fig)
    print("assembled fig_C_D from fig_C1 + fig_C2 + fig_D (no recomputation beyond their compute())")


if __name__ == "__main__":
    main()

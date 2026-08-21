"""Fig A (horizontal) — the same headline figure laid out side by side: ranking left, panels right.

ONE script, ONE figure: figures_v2/fig_A_horizontal.png / .pdf

THIS IS A SECOND LAYOUT OF THE SAME NUMBERS, NOT A SECOND ANALYSIS. figures/fig_A.py is the
stacked version — (a) full width, (b) as 2x3 beneath — and is what fits an A4 text block at 1:1.
This one is landscape / full-bleed at 11in and is kept for formats the stacked plate does not suit
(slides, a wide two-column spread, a poster). Both compose `fig_A1.draw()` and `fig_A2.draw_panel()`,
so every number, error bar and reference line is identical by construction; if the two ever
disagree, one of them failed to re-render.

Layout: the ranking fills the left column; the six canonical panels sit to the right as 2 rows x 3
columns, grouped by metric DIRECTION so a reader never has to re-orient between neighbours:

    row 1   MoleculeACE ↓   QM7 ↓    HIV ↑
    row 2   BACE ↑          Ames ↑   Tox21 ↑

WIDER THAN THE A4 TEXT BLOCK, ON PURPOSE. At 6.69in this layout had to run 8.1in down the page to
stay legible, which read as distorted. `save(..., wide=True)` records the exemption rather than
warning about it. The cost is real and is why it is not the main figure: a plate wider than the
text block is rotated or scaled down in LaTeX, and scaling down takes every font with it.

The ranking panel is drawn with `compact=True`: at ~37% of the plate it cannot carry the
annotation density it does standalone.

Run:  python3 -m figures.fig_A_horizontal
"""
from __future__ import annotations
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, check_font, row_ncol
import figures.fig_A1 as A1
import figures.fig_A2 as A2
from figures.arms import PANEL_ORDER

check_font()
INK = "#000000"

# right-hand grid, grouped by metric direction (both "lower is better" panels share row 1)
# The 2x3 arrangement is NOT PANEL_ORDER: it groups by metric so a reader scans one kind of number
# per row -- top row the two regression panels plus the rare-active screen, bottom row the three
# ROC-AUC panels. HIV took CBS's slot in the 2026-08-19 swap and this list was not updated with it,
# which is why fig_A stopped rendering (KeyError: 'CBS') rather than silently drawing a stale
# panel. The assert makes the next such swap fail at import instead of at draw time.
PANEL_GRID = [["MoleculeACE", "QM7", "HIV"], ["BACE", "Ames", "Tox21"]]
assert sorted(p for row in PANEL_GRID for p in row) == sorted(PANEL_ORDER), \
    f"fig_A's grid {PANEL_GRID} has drifted from arms.PANEL_ORDER {PANEL_ORDER}"


# DELIBERATELY WIDER THAN THE A4 TEXT BLOCK. At 6.69in this figure had to run 8.1in down the page
# to stay legible, which read as distorted. Set landscape / full-bleed instead: the 2x3 panel grid
# and the ranking sit side by side at a natural aspect. This is the ONE figure exempt from the page
# -width rule, and save(..., wide=True) records that rather than warning about it.
WIDTH = 11.0


def build(height=5.5, left_frac=0.375):
    fig = plt.figure(figsize=(WIDTH, height))
    # wspace is generous: the right-hand panels carry y-labels that would otherwise land on top
    # of the ranking panel. bottom reserves a band for BOTH legends.
    outer = fig.add_gridspec(1, 2, width_ratios=[left_frac, 1 - left_frac], wspace=0.10,
                             left=0.125, right=0.995, top=0.945, bottom=0.175)

    axL = fig.add_subplot(outer[0, 0])
    A1.draw(axL, compact=True)
    # Sentence case, matching every other title in the set ("Supervised: permuted targets",
    # "Lift by similarity group", "Transfer vs chemical similarity") and standard journal style.
    axL.set_title("Mean rank, four suites equally weighted", fontsize=FS["title"], fontweight="bold",
                  color=INK, pad=6)
    axL.text(-0.34, 1.012, "a", transform=axL.transAxes, fontsize=FS["panel_tag"],
             fontweight="bold", va="bottom", ha="left", color=INK)

    gs = outer[0, 1].subgridspec(2, 3, hspace=0.38, wspace=0.30)
    right_axes = []
    for r, row in enumerate(PANEL_GRID):
        for c, p in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            A2.draw_panel(ax, p, compact=True)
            right_axes.append(ax)
            if (r, c) == (0, 0):
                ax.text(-0.17, 1.075, "b", transform=ax.transAxes, fontsize=FS["panel_tag"],
                        fontweight="bold", va="bottom", ha="left", color=INK)

    # Each key is CENTRED on the x-extent of the half it describes and sits just under that half's
    # x-axis, rather than being parked in the figure's bottom corners. Centres are measured from
    # the realised axes positions, so they track any later change to the gridspec ratios.
    fig.canvas.draw()
    lp = axL.get_position()
    rp = [a.get_position() for a in right_axes]
    r_x0, r_x1 = min(p.x0 for p in rp), max(p.x1 for p in rp)
    r_bot = min(p.y0 for p in rp)

    _h1 = A1.suite_handles(with_dagger=True)
    fig.legend(handles=_h1, loc="upper center",
               bbox_to_anchor=((lp.x0 + lp.x1) / 2, lp.y0 - 0.085),
               ncol=row_ncol(_h1), frameon=False, fontsize=FS["legend"], handletextpad=0.4,
               labelspacing=0.3, columnspacing=1.2, borderpad=0.0, labelcolor=INK)
    _h2 = A2.legend_handles()
    fig.legend(handles=_h2, loc="upper center",
               bbox_to_anchor=((r_x0 + r_x1) / 2, r_bot - 0.042),
               # 12 handles. ONE ROW IS THE DEFAULT AND IT DOES NOT FIT HERE, measured rather
               # than assumed: one row took this plate from 10.85in to 13.96in, past even a
               # landscape A4 text block. rows=3 balances 12 into 4x3 -- the same three rows the
               # old ncol=5 produced, but even instead of 5/5/2.
               ncol=row_ncol(_h2, rows=3), frameon=False, fontsize=FS["legend"], handlelength=1.5,
               handletextpad=0.5, labelspacing=0.3, columnspacing=1.1, borderpad=0.0,
               labelcolor=INK)
    return fig


def main():
    fig = build()
    save(fig, "fig_A_horizontal", wide=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

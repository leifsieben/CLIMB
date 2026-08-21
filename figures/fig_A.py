"""Fig A — the headline result in one figure: overall standing (a) above the six panels (b).

ONE script, ONE figure: figures_v2/fig_A.png / .pdf

Layout: STACKED (user 2026-08-20). (a) is the 65-dataset ranking across the full text-block width;
(b) is the six canonical panels as 2x3 beneath it, grouped by metric DIRECTION so a reader never
has to re-orient between horizontal neighbours:

    row 1   MoleculeACE ↓   QM7 ↓    HIV ↑
    row 2   BACE ↑          Ames ↑   Tox21 ↑

This was side-by-side at 11in wide — landscape, the one figure exempt from the page-width rule.
The exemption cost more than it bought: a plate wider than the text block is either rotated or
scaled down in LaTeX, and scaling down takes every font with it, so the authored point sizes stop
being the sizes on the page. Stacked, the plate is 6.7 x 9.3in and goes in at 1:1.

Nothing is recomputed here. This composes `fig_A1.draw()` and `fig_A2.draw_panel()`, so the
numbers, error bars and reference lines are by construction identical to the standalone figures —
the same arrangement fig_C+D uses for C1/C2/D.

The ranking panel is drawn with `compact=True`, which now costs only the per-point value labels:
stacked, it has MORE horizontal room than it had at 37.5% of a landscape plate.

Run:  python3 -m figures.fig_A
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


# STACKED, AND THEREFORE BACK INSIDE THE A4 TEXT BLOCK (user 2026-08-20).
#
# This was side-by-side at 11in wide -- landscape / full-bleed, the one figure exempt from the
# page-width rule -- because the ranking panel is 22 rows tall and a full-width band of 22 rows
# plus a 2x3 grid beneath it is most of a page. The exemption cost more than it bought: a plate
# wider than the text block is either rotated or scaled down in LaTeX, and scaling down takes every
# font with it, so the authored point sizes stop being the sizes on the page.
#
# Stacked, (a) spans the full text-block width and (b) sits under it as 2x3. The ranking gets MORE
# horizontal room than it had at 37.5% of a landscape plate, not less, so `compact` now only drops
# the per-point value labels rather than also fighting for width.
WIDTH = STYLE["col2"]
HEIGHT = 9.3                      # measured to sit inside A4's ~10.1in text height with margin


def build(height=HEIGHT, top_frac=0.62):
    """(a) the ranking, full width; (b) the six panels as 2x3 beneath it.

    `top_frac` is the ranking's share of the vertical space. It needs the larger share: 22 model
    rows against 2 rows of panels.
    """
    fig = plt.figure(figsize=(WIDTH, height))
    # hspace is generous because (a)'s legend sits in the gap between the two halves rather than
    # at the foot of the figure -- a key three panels away from the marks it explains is a key the
    # reader has to hold in their head.
    outer = fig.add_gridspec(2, 1, height_ratios=[top_frac, 1 - top_frac], hspace=0.20,
                             left=0.185, right=0.995, top=0.972, bottom=0.075)

    axT = fig.add_subplot(outer[0, 0])
    A1.draw(axT, compact=True)
    # Sentence case, matching every other title in the set ("Supervised: permuted targets",
    # "Lift by similarity group", "Transfer vs chemical similarity") and standard journal style.
    axT.set_title("Mean rank, four suites equally weighted", fontsize=FS["title"],
                  fontweight="bold", color=INK, pad=6)

    gs = outer[1, 0].subgridspec(2, 3, hspace=0.42, wspace=0.32)
    bot_axes = []
    for r, row in enumerate(PANEL_GRID):
        for c, p in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            A2.draw_panel(ax, p, compact=True)
            bot_axes.append(ax)

    # Each key is centred on the half it describes and sits just under that half's x-axis, rather
    # than being parked at the foot of the plate. Positions are measured from the REALISED axes, so
    # they track any later change to the gridspec ratios instead of being re-tuned by hand.
    fig.canvas.draw()
    tp = axT.get_position()
    # PANEL TAGS IN FIGURE COORDINATES, at ONE x. Placed per-axes they were not aligned: an
    # offset given in axes-fraction is a different absolute distance for a full-width panel than
    # for a third-width one, so "a" sat at the plate edge and "b" a centimetre inside it.
    bp = [a.get_position() for a in bot_axes]
    b_x0, b_x1 = min(q.x0 for q in bp), max(q.x1 for q in bp)
    b_bot = min(q.y0 for q in bp)
    TAG_X = 0.012
    for tag, y in (("a", tp.y1), ("b", max(q.y1 for q in bp))):
        fig.text(TAG_X, y + 0.006, tag, fontsize=FS["panel_tag"], fontweight="bold",
                 va="bottom", ha="left", color=INK)

    _h1 = A1.suite_handles(with_dagger=True)
    fig.legend(handles=_h1, loc="upper center",
               bbox_to_anchor=((tp.x0 + tp.x1) / 2, tp.y0 - 0.038),
               ncol=row_ncol(_h1), frameon=False, fontsize=FS["legend"], handletextpad=0.4,
               labelspacing=0.3, columnspacing=1.2, borderpad=0.0, labelcolor=INK)
    _h2 = A2.legend_handles()
    fig.legend(handles=_h2, loc="upper center",
               bbox_to_anchor=((b_x0 + b_x1) / 2, b_bot - 0.030),
               # 12 handles. One row is the default and does not fit, and neither does two:
               # 6 columns rendered 7.49in against a 6.69in text block, i.e. the legend and not
               # the panels was setting the plate width. rows=3 -> 4 columns, measured.
               ncol=row_ncol(_h2, rows=3), frameon=False, fontsize=FS["legend"], handlelength=1.5,
               handletextpad=0.5, labelspacing=0.3, columnspacing=1.1, borderpad=0.0,
               labelcolor=INK)
    return fig


def main():
    fig = build()
    # NOT wide= any more: the plate is inside the text block, so the page-width
    # check should police it like every other figure rather than be waived.
    save(fig, "fig_A")
    plt.close(fig)


if __name__ == "__main__":
    main()

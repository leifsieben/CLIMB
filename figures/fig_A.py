"""Fig A — the headline result in one figure: overall standing (left) + the six panels (right).

ONE script, ONE figure: figures_v2/fig_A.png / .pdf

Layout: fig_A1's 66-dataset ranking fills the left column; fig_A2's six canonical panels sit to the
right as 3 rows x 2 columns. Rows are grouped by metric DIRECTION so a reader never has to re-orient
between neighbours:

    row 1   MoleculeACE ↓   QM7 ↓   CBS ↑
    row 2   BACE ↑          Ames ↑   Tox21 ↑

The two lower-is-better panels sit side by side at the start of row 1, so a reader never meets a
direction flip between horizontal neighbours.

Nothing is recomputed here. This composes `fig_A1.draw()` and `fig_A2.draw_panel()`, so the numbers,
error bars and reference lines are by construction identical to the standalone figures — the same
arrangement fig_C+D uses for C1/C2/D.

The ranking panel is drawn with `compact=True`: no per-point value labels and smaller markers,
because at ~40% of the text block it cannot carry the annotation density it does standalone.

Run:  python3 -m figures.fig_A
"""
from __future__ import annotations
import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, check_font
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
    axL.set_title("Mean rank across all datasets", fontsize=FS["title"], fontweight="bold",
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

    fig.legend(handles=A1.suite_handles(), loc="upper center",
               bbox_to_anchor=((lp.x0 + lp.x1) / 2, lp.y0 - 0.085),
               ncol=4, frameon=False, fontsize=FS["legend"], handletextpad=0.4,
               labelspacing=0.3, columnspacing=1.2, borderpad=0.0, labelcolor=INK)
    fig.legend(handles=A2.legend_handles(), loc="upper center",
               bbox_to_anchor=((r_x0 + r_x1) / 2, r_bot - 0.042),
               ncol=5, frameon=False, fontsize=FS["legend"], handlelength=1.5,
               handletextpad=0.5, labelspacing=0.3, columnspacing=1.1, borderpad=0.0,
               labelcolor=INK)
    return fig


def main():
    fig = build()
    save(fig, "fig_A", wide=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

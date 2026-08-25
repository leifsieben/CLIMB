"""Fig C+D -- the assembled similarity & transfer figure: C1 + C2 + D in one canvas, panels a-f.

ONE script, ONE figure: figures_v2/fig_C+D.png / .pdf

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

Run:  python3 -m figures.fig_C_plus_D

PANEL SET — FULLY MIGRATED to the canonical six as of 2026-08-19. The last blocker was the
full-corpus MoleculeACE similarity for fig_C1's x-axis.

ONE PANEL IS SHORT, and it is a data gap rather than a choice: fig_C2 and fig_D draw FIVE of the
six because Tox21 is withheld. The ablation wave's per-molecule Tox21 dumps are pre-fix (93,876
rows rather than the masked 77,864), so they cannot be re-scored from disk, and re-evaluating them
in a different environment left a ~0.0075 offset — 15-40% of the Tox21 lifts being measured. See
figures.sixpanel.CORRECTION_TASKS for why a correction, unlike a unit convention, cannot simply be
applied to both sides.
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

    # RE-LAID AT THE A4 TEXT BLOCK (user 2026-08-19). It was 8.9in -- 33% over -- which LaTeX
    # scales to 0.75x at \includegraphics[width=\textwidth], so its fonts printed a quarter
    # smaller than every other figure in the set while every script set the same points.
    #
    # This costs almost nothing in printed panel size, which is the part that looks like a
    # trade-off and is not: a 2.4in panel inside an 8.9in figure already arrives on the page at
    # 2.4 x 6.69/8.9 = 1.80in. Laying it out at 6.69 natively gives ~1.75in panels at 1:1. The
    # earlier "cramped at 6.69" finding (2026-08-17) was at the OLD height; the fix is to buy the
    # room back vertically rather than horizontally, since fonts now render at their authored size.
    # ONE ROW, FOUR PANELS (Leif 2026-08-23). The two-row a-f assembly was overloaded; this keeps
    # the four panels that carry the argument and drops b (the C1 trend, whose claim panel a
    # already states) and d (the D bar summary, which the e matrix contains in full).
    #
    # Order is the argument's order: molecular similarity for the UNSUPERVISED objective, then for
    # the SUPERVISED one, then task similarity twice. The two group titles sit ABOVE their pair
    # rather than rotated at the left edge, so the reader gets the grouping before the panels
    # instead of after.
    # THREE PANELS (Leif 2026-08-23). The a-f two-row assembly was overloaded; four was still
    # tight. What survives is one panel per question: molecular similarity for the UNSUPERVISED
    # objective, the same for the SUPERVISED one, then the task-similarity matrix.
    #
    # Dropped, and why each is safe to drop: b (the C1 trend) restated panel a's claim as a
    # continuous version of the same lift; d (the D bar summary) is the row-means of the matrix
    # that is still drawn; f (the dense-vs-sparse slope) is one cut of that same matrix. Every
    # dropped panel remains in its standalone figure, so nothing is lost from the record.
    # A4 TEXT BLOCK EXACTLY (Leif 2026-08-23). STYLE["col2"] is that width, so the figure arrives
    # at 1:1 under \includegraphics[width=\textwidth] and its fonts print at the size every other
    # script authored them at. At 7.0in it was 5% over and LaTeX silently scaled it down.
    fig = plt.figure(figsize=(STYLE["col2"], 3.05))
    # The gridspec span is deliberately narrower than the canvas: the two-pass balancer below
    # keeps axes WIDTHS and only repositions them, so the room for the gaps has to exist before
    # it runs. At the full span the panels packed to a 0.011 gap and read as one block.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.18, 1.30], wspace=0.46,
                          left=0.078, right=0.912, top=0.775, bottom=0.325)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # fig_C1.draw and fig_D.draw render their panels as a set, and those signatures are shared with
    # the standalone figures -- changing them to render subsets would put this layout's needs into
    # scripts that answer to a different one. The unwanted panels go to a scratch canvas that is
    # closed immediately, which costs one wasted render and keeps the shared entry points intact.
    scratch = plt.figure(figsize=(4, 6))
    unused = [scratch.add_subplot(3, 1, i) for i in (1, 2, 3)]

    fig_C1.draw(ax_a, unused[0], d1, tags=("a", ""), compact=True)
    fig_C2.draw(ax_b, d2, tag="b", compact=True)
    fig_D.draw(unused[1], ax_c, unused[2], d3, tags=("", "c", ""), compact=True)
    plt.close(scratch)

    # TWO-PASS LAYOUT, because these three panels carry very different decorations and a uniform
    # wspace cannot balance them. Measured on the one-pass version: panels b and c OVERLAPPED by
    # 0.027 of the figure width -- b's x-label ran past its axes into c's row labels -- while the
    # left and right margins were 0.022 and 0.012. That is what "c is a bit off" was.
    #
    # So: draw once, measure each panel's DRAWN extent (tight bbox, decorations included), then
    # reposition the axes so the two outer margins are equal and the two gaps are equal. Deriving
    # the geometry from what was actually rendered means it stays balanced when a label changes
    # length, which a hand-tuned wspace does not.
    def _bbs():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        return [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                for ax in (ax_a, ax_b, ax_c)]

    MARGIN = 0.014
    panels = (ax_a, ax_b, ax_c)
    bbs = _bbs()
    drawn_total = sum(b.x1 - b.x0 for b in bbs)
    gap = (1.0 - 2 * MARGIN - drawn_total) / (len(bbs) - 1)

    # ATTACHED AXES MOVE WITH THEIR PANEL. fig_D builds the matrix colourbar with
    # colorbar(im, ax=axM), which steals space from axM and places its own axes at CREATION time --
    # matplotlib keeps no link afterwards, so set_position on the parent leaves the bar behind.
    # First pass without this put the colourbar under the wrong x and the matrix's rotated column
    # labels landed on top of it. Anything that is not one of the three panels is claimed by
    # whichever panel currently spans its centre, and is shifted by the same dx.
    others = [ax for ax in fig.axes if ax not in panels]
    owner = {}
    for ax in others:
        cx = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        for i, pax in enumerate(panels):
            pp = pax.get_position()
            if pp.x0 - 0.05 <= cx <= pp.x1 + 0.05:
                owner.setdefault(ax, i)
                break

    cursor, dxs = MARGIN, []
    for ax, bb in zip(panels, bbs):
        pos = ax.get_position()
        pad_left = pos.x0 - bb.x0            # y-label, row labels: whatever sits left of the axes
        new_x0 = cursor + pad_left
        dxs.append(new_x0 - pos.x0)
        ax.set_position([new_x0, pos.y0, pos.width, pos.height])
        cursor += (bb.x1 - bb.x0) + gap
    for ax, i in owner.items():
        pos = ax.get_position()
        ax.set_position([pos.x0 + dxs[i], pos.y0, pos.width, pos.height])

    # AND DROP IT CLEAR OF THE ROTATED COLUMN LABELS, by the measured overlap rather than by a
    # guessed pad. fig_D's colorbar pad is tuned for its own standalone layout, where the matrix is
    # taller; in this row the rotated task names reach 0.059 of the figure BELOW the bar's top and
    # printed straight through it. Measuring means this stays correct if the label set changes.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, i in owner.items():
        if i != 2:
            continue
        lab_bottom = min((t.get_window_extent(r).transformed(fig.transFigure.inverted()).y0
                          for t in ax_c.get_xticklabels() if t.get_text()), default=None)
        if lab_bottom is None:
            continue
        pos = ax.get_position()
        overlap = pos.y1 - lab_bottom
        if overlap > 0:
            drop = overlap + 0.030
            ax.set_position([pos.x0, pos.y0 - drop, pos.width, pos.height])
            print(f"  fig_C+D: colourbar dropped {drop:.3f} clear of the rotated column labels")
    print(f"  fig_C+D: balanced to margin {MARGIN:.3f}, equal gaps {gap:.3f}; "
          f"{len(owner)} attached axes moved with their panel")

    def _span(axes):
        xs = []
        for ax in axes:
            bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            xs += [bb.x0, bb.x1]
        return (min(xs) + max(xs)) / 2.0
    fig.canvas.draw()

    fig.text(_span([ax_a, ax_b]), 0.950, "Molecular Similarity", ha="center", va="center",
             fontsize=FS["panel_tag"], fontweight="bold")
    fig.text(_span([ax_c]), 0.950, "Task Similarity", ha="center", va="center",
             fontsize=FS["panel_tag"], fontweight="bold")

    save(fig, "fig_C+D")
    plt.close(fig)
    print("assembled fig_C+D from fig_C1 + fig_C2 + fig_D (no recomputation beyond their compute())")


if __name__ == "__main__":
    main()

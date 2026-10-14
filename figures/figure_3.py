from __future__ import annotations

import matplotlib.pyplot as plt

from figures.style import STYLE, FS, save, title, check_font
from figures import corpus_similarity, supervised_similarity, task_similarity

check_font()


def main():
    d1 = corpus_similarity.compute()
    d2 = supervised_similarity.compute()
    d3 = task_similarity.compute()


    fig = plt.figure(figsize=(STYLE["col2"], 3.05))


    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.18, 1.30], wspace=0.46,
                          left=0.078, right=0.912, top=0.775, bottom=0.325)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])


    scratch = plt.figure(figsize=(4, 6))
    unused = [scratch.add_subplot(3, 1, i) for i in (1, 2, 3)]

    corpus_similarity.draw(ax_a, unused[0], d1, tags=("a", ""), compact=True)
    supervised_similarity.draw(ax_b, d2, tag="b", compact=True)
    task_similarity.draw(unused[1], ax_c, unused[2], d3, tags=("", "c", ""), compact=True)
    plt.close(scratch)


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
        pad_left = pos.x0 - bb.x0
        new_x0 = cursor + pad_left
        dxs.append(new_x0 - pos.x0)
        ax.set_position([new_x0, pos.y0, pos.width, pos.height])
        cursor += (bb.x1 - bb.x0) + gap
    for ax, i in owner.items():
        pos = ax.get_position()
        ax.set_position([pos.x0 + dxs[i], pos.y0, pos.width, pos.height])


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

    save(fig, "figure_3")
    plt.close(fig)
    print("assembled fig_C+D from fig_C1 + fig_C2 + fig_D (no recomputation beyond their compute())")


if __name__ == "__main__":
    main()

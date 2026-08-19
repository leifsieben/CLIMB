"""Fig E+F -- the assembled objective figure: corrupted pretraining (E) + feature redundancy (F).

ONE script, ONE figure: figures_v2/fig_E+F.png / .pdf   (panels a-h)

Layout (user request 2026-08-19): E's two panels STACKED in a narrow left column, F's six canonical
panels in a 2x3 block on the right. Putting them on one canvas is not just page economy -- the two
halves answer the same question from opposite directions:

  left  (a, b)   REMOVE the chemistry from pretraining and see what survives
  right (c-h)    ADD the pretrained representation to classical features and see what it contributes

  a  E supervised    real descriptor targets vs targets permuted across the batch
  b  E unsupervised  the corpus-degradation ladder (shuffled / bigram / unigram / Wikipedia)
  c-h  F             ECFP+desc vs CLIMB alone vs CLIMB+desc vs CLIMB+desc+fp, per canonical panel

Read together: (a-b) say the unsupervised benefit is largely NOT chemistry-specific -- Wikipedia
with zero molecules in it is positive on 5 of 6 panels -- while the supervised benefit IS the
molecule->label correspondence, since permuting the targets lands below the untrained floor
everywhere. Then (c-h) say that whatever either objective learned adds nothing on top of ECFP+desc
on any canonical panel. The bars in the right block are narrowed (0.58 vs 0.72) so six panels fit
without the whiskers colliding.

No analysis code lives here: fig_E and fig_F expose their table/compute + draw entry points and this
script only arranges axes, so the standalone figures and this assembly can never drift apart --
the same contract fig_C+D uses.

Run:  python3 -m figures.fig_E_plus_F
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from figures.style import STYLE, FS, save, check_font
from figures.arms import PANEL_ORDER
import figures.fig_E as E
import figures.fig_F as F

check_font()


# the standalone subtitles ("Unsupervised: degraded corpus statistics") run past a narrow axes and
# collide with the F block, so the assembled figure uses short forms and carries the full wording
# in the caption.
SHORT = {"supervised": "Supervised", "unsupervised": "Unsupervised"}


def main():
    dE = pd.read_csv(E.TABLE)
    dF = F.compute()

    fig = plt.figure(figsize=(STYLE["col2"], 4.95))
    # left column carries two tall panels with six x-labels each, so it needs more width per panel
    # than the right block; 1.35 : 1 : 1 : 1 keeps "MoleculeACE" legible on the E axes.
    # 50:50 (user 2026-08-19) -- the E column equals the three F columns combined, so the two
    # halves of the figure carry equal visual weight. F's bars narrow to 0.42 to suit.
    # TWO grids at explicit figure coordinates (user 2026-08-19). Nested subgridspecs could not
    # express "F's panels stop early and its legend sits in the gap underneath", because the legend
    # is a figure artist and does not participate in the grid. So: E spans the full height; F's
    # panels stop at F_BOT and their one-row legend is centred just below, which makes
    # "F panels + F legend" and "E column" finish at the same depth.
    F_BOT, TOP = 0.215, 0.930
    # left/bottom are generous because the E panels carry a y-label AND rotated tick labels; a
    # tight bbox that clips them pushes the SAVED width past the 6.69in text block.
    gsE = GridSpec(2, 1, figure=fig, hspace=0.55,
                   left=0.098, right=0.470, top=TOP, bottom=0.150)
    gsF = GridSpec(2, 3, figure=fig, wspace=0.46, hspace=0.55,
                   left=0.570, right=0.995, top=TOP, bottom=F_BOT)

    ylims = {panel: E._lim(dE[dE.panel == panel], pad_hi=hi)
             for (panel, _, _, _), hi in zip(E.PANELS, (0.55, 0.95))}
    for row, (panel, _tag, subtitle, series) in enumerate(E.PANELS):
        ax = fig.add_subplot(gsE[row, 0])
        E.draw(fig, ax, dE[dE.panel == panel], series, "ab"[row], SHORT[panel], ylims[panel],
               compact=True)
        ax.set_ylabel("Lift over no pretrain, frozen", fontsize=FS["annot"])

    ylims = F.shared_ylims(dF)          # panels on one metric share one y-range
    tags = "cdefgh"
    for k, p in enumerate(PANEL_ORDER):
        ax = fig.add_subplot(gsF[k // 3, k % 3])
        F.draw_panel(ax, dF, p, compact=True, tag=tags[k], fig=fig, ylims=ylims)

    # directly beneath the F block, one row, anchor dropped
    fig.legend(handles=F.legend_handles(skip_anchor=True), loc="upper center",
               bbox_to_anchor=(0.783, F_BOT - 0.075), ncol=3, fontsize=FS["legend"],
               handletextpad=0.5, columnspacing=1.5, borderpad=0.0, frameon=False)
    save(fig, "fig_E+F")
    plt.close(fig)
    print("assembled fig_E+F from fig_E + fig_F (no recomputation beyond their own entry points)")


if __name__ == "__main__":
    main()

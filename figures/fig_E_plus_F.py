"""Fig E+F -- the assembled objective figure: corrupted pretraining (E) + feature redundancy (F).

ONE script, ONE figure: figures_v2/fig_E+F.png / .pdf   (panels a-h)

LAYOUT -- STACKED, not side by side (user 2026-08-19: "neither one is very readable right now...
they would need to be bigger"). The previous version split the 6.69in text block LEFT/RIGHT, which
gave E's two panels 2.4in each and F's six panels 1.15in each. F's panels in particular could not
carry four tick labels and a y-axis at that width, so the labels were rotated 40 deg and the tick
font dropped to 6pt. Stacking recovers the full text-block width for BOTH halves at the cost of
page height only:

    E:  2 panels, 2.5in and 3.6in wide   (was 2.4in each)   -- panel b's 30 bars stop colliding
    F:  6 panels, ~1.8in wide each       (was 1.15in)       -- +57%, enough for HORIZONTAL x labels

Nothing else about the two halves changed. They still belong on one canvas because they answer the
same question from opposite directions:

  top    (a, b)   REMOVE the chemistry from pretraining and see what survives
  bottom (c-h)    ADD the pretrained representation to classical features and see what it contributes

  a  E supervised    real descriptor targets vs targets permuted across the batch
  b  E unsupervised  the corpus-degradation ladder (shuffled / bigram / unigram / Wikipedia)
  c-h  F             ECFP+desc vs CLIMB alone vs CLIMB+desc vs CLIMB+desc+fp, per canonical panel

Read together: (a-b) say the unsupervised benefit is largely NOT chemistry-specific -- Wikipedia
with zero molecules in it is positive on 5 of 6 panels -- while the supervised benefit IS the
molecule->label correspondence, since permuting the targets lands below the untrained floor
everywhere. Then (c-h) say that whatever either objective learned adds nothing on top of ECFP+desc
on any canonical panel.

Because each half now spans the full width, E is drawn in its STANDALONE form (compact=False): the
full subtitles fit, and its legends go back to one column at the standard legend size instead of
the two-column 6pt squeeze the narrow column forced.

No analysis code lives here: fig_E and fig_F expose their table/compute + draw entry points and this
script only arranges axes, so the standalone figures and this assembly can never drift apart --
the same contract fig_C+D uses.

Run:  python3 -m figures.fig_E_plus_F
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from figures.style import STYLE, FS, save, check_font, row_ncol
from figures.arms import PANEL_ORDER, ARMS
import figures.fig_E as E
import figures.fig_F as F

check_font()


def main():
    dE = pd.read_csv(E.TABLE)
    dF = F.compute()

    # 6.69 x 6.6in. The width is the A4 text block (never hard-coded, see style.col2); the height
    # is set by what the two halves need: ~1.75in of axes for E's row and ~1.4in per row for F's
    # two. That lands well inside A4's 10.1in text height, so the figure and its caption share a
    # page.
    # ONE RIGHT EDGE FOR BOTH HALVES (user 2026-08-20: "have the width of E and F be the same").
    # F used to stop at 0.838 so its legend could sit in the gutter beside it; that made F's six
    # panels 1.78in against E's 2.5/3.6in and, more to the point, put the two halves of one figure
    # on two different widths. The legend moves under F instead. That is the trade the previous
    # comment named and declined -- it costs page height for the legend row and buys back the
    # alignment.
    RIGHT = 0.995            # E, F, and the page
    F_TOP, F_BOT = 0.600, 0.145
    fig = plt.figure(figsize=(STYLE["col2"], 6.45))
    # width_ratios [1.0, 1.45] is fig_E's own: panel b carries five series per group and panel a
    # only two, so b needs the extra width to keep the ladder legible.
    # ONE right edge for both halves. The legend is right-aligned to the SAME number, which is
    # what "aligned with the right side of E" means once E and F are stacked and share it.
    gsE = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.45], wspace=0.24,
                   left=0.082, right=RIGHT, top=0.962, bottom=0.735)
    gsF = GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.40,
                   left=0.082, right=RIGHT, top=F_TOP, bottom=F_BOT)

    ylims = {panel: E._lim(dE[dE.panel == panel]) for panel, _, _, _ in E.PANELS}
    for col, (panel, tag, subtitle, series) in enumerate(E.PANELS):
        ax = fig.add_subplot(gsE[0, col])
        E.draw(fig, ax, dE[dE.panel == panel], series, tag, subtitle, ylims[panel])
        ax.set_ylabel("Lift over " + ARMS["random_encoder"]["label"], fontsize=FS["label"])

    ylims = F.shared_ylims(dF)          # panels on one metric share one y-range
    tags = "cdefgh"
    for k, p in enumerate(PANEL_ORDER):
        ax = fig.add_subplot(gsF[k // 3, k % 3])
        # xrot=0: at ~1.8in per panel the four short labels fit horizontally, which is the whole
        # point of the restack.
        F.draw_panel(ax, dF, p, compact=True, tag=tags[k], fig=fig, ylims=ylims, xrot=0, bw=0.62)

    # HORIZONTAL, UNDER F, spanning its full width. The anchor entry is dropped -- it is the
    # first bar in every panel, tick-labelled "ECFP+d", and is the dotted reference line, so it
    # needs no swatch. Four entries fit one row comfortably at the full text-block width, which is
    # the house default (figures/style.row_ncol) and what the gutter version could never do.
    _h = F.legend_handles(skip_anchor=True, wrap=False)
    fig.legend(handles=_h, loc="upper center", bbox_to_anchor=(0.5, F_BOT - 0.072),
               ncol=row_ncol(_h), fontsize=FS["legend"], handletextpad=0.5, columnspacing=1.4,
               labelspacing=0.3, borderpad=0.0, frameon=False)
    save(fig, "fig_E+F")
    plt.close(fig)
    print("assembled fig_E+F from fig_E + fig_F (no recomputation beyond their own entry points)")


if __name__ == "__main__":
    main()

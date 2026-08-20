"""SI Fig a — do you need to train the models end-to-end on the downstream data?

ONE script, ONE figure: figures_v2/SI_fig_a.png / .pdf

The same pretrained encoder used two ways at FULL downstream data: frozen (encoder fixed, probe
trained on the labels) versus end-to-end (whole network fine-tuned). Three encoders — `unsupervised`,
`supervised, desc`, and the external comparator `CheMeleon` — on each of the six canonical panels.
Thirty-six points, no holes.

ONE WAVE, ONE ESTIMATOR (rebuilt 2026-08-20). Both properties were missing and both mattered:

  WAVE. The figure was built from two waves — mainline on MoleculeACE/Ames/HIV, label-efficiency at
  its 100% fraction on BACE/Tox21/QM7. That was a historical accident: when it was chosen the
  end-to-end CLIMB arms had no MolNet runs. They have them now, so every point comes from the
  mainline wave. The offsets this removes are large — the same ECFP4+desc anchor reads 0.8712 on
  BACE in one wave and 0.7836 in the other — and it fills three real holes, since CheMeleon was
  frozen-only on BACE and Tox21 and absent from QM7.

  ESTIMATOR. Error bars were "±1 SD of that panel's replicate unit", and the replicate unit was not
  the same thing for every arm — pretraining-seed spread where an arm has three pretrainings,
  head-seed spread for CheMeleon, which has one pretraining by construction. Different estimands on
  one axis look like a precision difference and are not. Every interval is now a 95% resampling
  interval from the same file fig_A2 draws from, and every arm WITHIN a panel shares one method:
  scaffold cluster bootstrap on the MolNet panels, target cluster bootstrap on MoleculeACE (whose
  value is a macro-mean over 30 separate tasks, so targets are the resampling unit), and an
  analytic interval on Ames, whose test labels Polaris withholds. The method is panel-shaped
  because the data is; what matters is that it is identical for every arm you are asked to compare,
  which is within a panel. Points come from the same file as their intervals, so the two can never
  describe different estimators.

  Intervals are ASYMMETRIC because the bootstrap distribution is, and they are drawn that way.

WHICH CHEMELEON. The frozen half is the XGBOOST probe. The paper reports exactly two CheMeleon
models — frozen+XGBoost and end-to-end-from-foundation (Leif 2026-08-20) — and this is the same
convention fig_A1 uses: each representation at the head that suits it, a preference SI fig f
measures as a property in its own right. The honest consequence, stated in the caption: CheMeleon's
line changes head between its ends (XGBoost probe → fine-tuned D-MPNN) while the CLIMB lines do
not. Reporting its MLP probe instead would match heads at the cost of drawing a configuration the
paper never otherwise mentions, and would understate the arm by 0.185 macro RMSE on MoleculeACE.

WHAT THE FIGURE SAYS, AND IT IS WEAKER THAN THE PREVIOUS VERSION CLAIMED. End-to-end is ahead in 12
of the 18 cells, but NOT ONE of the eighteen frozen/end2end interval pairs is disjoint. The old
figure reported 8 of 12 cells "clearing the combined SD"; that bar was a spread over seed
replicates, which measures how reproducible a number is and not how well the test set pins it down.
A cluster bootstrap over test scaffolds includes the sampling variation the seed SD omits, and it
is several times wider. Read this figure as "end-to-end is usually ahead, and the data cannot
resolve any single one of these differences", not as a set of wins.

CAVEAT ON THAT, IN THE OPPOSITE DIRECTION: overlapping marginal intervals are a CONSERVATIVE test.
Frozen and end2end are scored on the SAME molecules, so a paired bootstrap — resample the scaffolds
once, take the difference within the resample — would cancel the shared test-set draw and give a
much tighter interval on the delta. That is the correct test for this figure's question and it is
not what is drawn here. The non-overlap statement above is therefore a floor on what can be
claimed, not a verdict.

CHEMELEON IS THE LINE THAT CHANGED MOST, because it is now the XGBoost probe. With the MLP probe
its frozen end sat far below its fine-tuned end on every panel, which made it look like the clearest
case for fine-tuning in the paper. At the head that actually suits it, frozen is AHEAD on four of
six panels — HIV by 0.081 NEF1, Ames by 0.025 ROC-AUC, QM7 by 4.2 RMSE, BACE by 0.004 — and behind
only on MoleculeACE and Tox21. The apparent case for fine-tuning that arm was largely a case
against its MLP probe.

Compare frozen vs end2end WITHIN a panel, and across encoders within a panel — both are like-for-
like now. Panels still carry different metrics, so never compare a value in one panel against a
value in another.

Data: figure_data/SI_fig_a/SI_fig_a_e2e_need.csv, built by scripts/build_SI_fig_a_table.py from
figure_data/six_panel/a2_errorbars.csv.

Run:  python3 scripts/a2_bootstrap_errorbars.py       # if the intervals are stale
      python3 scripts/build_SI_fig_a_table.py && python3 -m figures.SI_fig_a
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol
from figures.arms import ARMS, PANELS, PANEL_ORDER, E2E_PAIRS, series_label
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_a" / "SI_fig_a_e2e_need.csv")

# A SLOPE plot, not bars (user request 2026-08-17): each encoder is ONE line joining its frozen
# value to its end2end value, so the only thing the reader has to judge is the DIRECTION and
# steepness of the change — which is the whole question — rather than comparing four bar heights.
# Colour = encoder family: blue = unsupervised, red = supervised, dense.
# (join key, arms.py key, legend label) -- all three from arms.py, never a literal. The join key
# must be byte-identical to what build_SI_fig_a_table.py wrote into the `encoder` column, and the
# two were separate literals until 2026-08-20, when arms.py's rename of sup_dense to "supervised,
# desc" left this file asking for "supervised, dense". The join matched nothing and the supervised
# line disappeared from every panel without any check firing.
# (series label, arm key used for COLOUR, legend label). All three derived from arms.py's
# E2E_PAIRS -- the same registry scripts/build_SI_fig_a_table.py writes the table from -- so the
# join key on both sides is one object. They were separate literals once and drifted: arms.py
# renamed sup_dense to "supervised, desc" while this file asked for "supervised, dense", the join
# matched nothing, and that encoder's line vanished from all six panels with nothing failing.
SERIES = [(series_label(fr), fr, series_label(fr)) for fr, _ in E2E_PAIRS]
PROBES = ["frozen", "end2end"]

# FAIL LOUDLY ON A DEAD JOIN KEY. A series whose key is absent from the table draws nothing and
# says nothing -- the panels stay populated by the other series, so no empty-panel check fires.
# Both sides now come from arms.py, so this should be unreachable; it is here because it was
# reachable once and cost the supervised encoder's line across all six panels.
_missing = [k for k, _, _ in SERIES if k not in set(DF.encoder)] if len(DF) else []
assert not _missing, (f"SI fig a: series key(s) {_missing} are not in the table's encoder column "
                      f"{sorted(set(DF.encoder))} -- rebuild with scripts/build_SI_fig_a_table.py")

# The classical anchor as a reference line, ON EVERY PANEL. One wave, so one source.
#
# This used to resolve per panel, because the figure was built from two waves and a mismatched
# anchor is worse than no anchor: the SAME ECFP4+desc features through the SAME XGBoost read 0.8712
# on BACE in the mainline wave and 0.7836 under label-efficiency -- 8.8 points, larger than the
# entire spread between arms on that panel -- so a mainline line on a label-efficiency panel drew
# the anchor roughly a protocol above where it belonged, and every model-to-anchor gap read off
# those panels was measuring the wave. The figure is single-wave now, so the resolver is gone
# rather than left as a branch that can never fire.
#
# The two waves differ ONLY in split construction: mainline is scaffold 5-fold CV,
# label-efficiency a single scaffold hold-out, with identical training-set sizes (1,210 on BACE).
# The hold-out is markedly harder, and harder FOR THE FINGERPRINT SPECIFICALLY -- the anchor lands
# below its own worst CV fold there while the frozen encoders land inside their fold range. That
# is a real, protocol-dependent result and it belongs with SI fig e, which is a label-efficiency
# figure; it is not something this figure can show while drawing one wave.
#
# METRIC IS MATCHED EXPLICITLY, not positionally: mainline_8M carries one row per (arm, panel), but
# reading `value` without checking `metric` is the failure mode that has cost this project a panel
# more than once, and audit check 15 exists because of it.
ANCHOR_ARM = "ecfp_desc"


def _anchor_values(protocols):
    """{panel: (value, source)} for the classical anchor, from the same wave as every point."""
    import csv as _csv
    del protocols                                   # single wave: nothing left to match on
    out = {}
    main = ROOT / "figure_data" / "six_panel" / "mainline_8M.csv"
    if main.exists():
        for r in _csv.DictReader(main.open()):
            if (r["arm"] == ANCHOR_ARM and r["panel"] in PANELS
                    and r["metric"] == PANELS[r["panel"]]["metric"]
                    and r["value"] not in ("", "nan")):
                out[r["panel"]] = (float(r["value"]), "mainline")
    return out


# "end2end" spelled out (user 2026-08-19: "e2e that is not commonly understood"). It does not fit
# horizontally under a ~1.1in panel, so the x tick labels are rotated instead of abbreviated --
# shortening to jargon to win space is the wrong trade.
XTICKS = ["frozen", "end2end"]


def main():
    # ONE PROTOCOL FOR THE WHOLE FIGURE, ASSERTED RATHER THAN RESOLVED PER PANEL.
    #
    # This used to be a per-panel resolver, because the table was built from two waves. It is not
    # any more -- every point comes from the mainline wave -- so the only thing left to do is
    # CHECK that, and fail if a future edit reintroduces a second wave without anyone noticing.
    #
    # The resolver it replaces was not academic. It was written as
    # `{r.panel: str(r.protocol) for r in DF.itertuples()}` -- last row per panel wins -- which
    # was fine until the CheMeleon rows were appended last and flipped every panel's protocol to
    # "mainline", which in turn drew the MAINLINE anchor on three label-efficiency panels. The
    # anchor gap there is 8.8 points on BACE, larger than the spread between arms.
    _waves = sorted(set(DF.protocol.astype(str))) if len(DF) else []
    assert len(_waves) <= 1, (f"SI fig a: the table mixes protocols {_waves}. Every point in this "
                             f"figure must come from one wave -- see the builder's docstring.")
    PROTO = {p: (_waves[0] if _waves else "") for p in PANEL_ORDER}
    ANCHOR = _anchor_values(PROTO)
    # 2x3 at FULL page width. One row of six was tried and reverted (user 2026-08-19: "too
    # extreme... they become super distorted") -- six panels across 6.69in leaves ~1.05in
    # each, taller than they are wide, which squashes the curves. 2x3 gives ~2.0in panels.
    # The height saving comes from tighter spacing and ONE shared x-axis label instead of
    # six, not from collapsing the grid. Width is ~3.5% over col2 because savefig("tight")
    # trims back to about the text block.
    _ls_kinds = []
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.3))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        g_all = DF[DF.panel == p]
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # Anchor first, so it is drawn even on a panel with no encoder data -- there the line is
        # the only content and it is what would make an empty panel worth printing at all.
        av = ANCHOR.get(p, (None, None))[0]
        if av is not None:
            ax.axhline(av, color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3, zorder=2)

        if g_all.empty:
            # Short enough to sit INSIDE a ~1.1in panel. The long form overran into both
            # neighbours' y-axis labels once the grid went to one row.
            ax.set_ylabel("")
            ax.text(0.5, 0.5, "end2end\nnot run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"] - 0.5, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            # DECLARED empty, so style.check_no_empty_panels passes it and fails on any panel
            # that is empty by accident instead. As of 2026-08-20 this fires for NO panel: the
            # builder refuses to write a table with a hole in it, so an empty panel here means the
            # figure and its table disagree about the panel set, which is worth crashing on rather
            # than drawing. Kept as a guard, not as a state the figure expects to be in.
            mark_empty(ax, f"{p}: no end2end run of a pretrained encoder on this panel")
            continue

        # EVERY SERIES IS SOLID (user 2026-08-20: "don't draw it dashed please"), and there is
        # nothing left for a dash to mark: the figure is single-wave, so no line here is drawn on
        # a protocol other than its panel's. The hook stays because a line style is the wrong
        # place to encode a caveat and the next person should have to delete this to add one.
        def _ls_for(enc_label):
            del enc_label
            return "-"

        # INTERVALS ARE ASYMMETRIC AND DRAWN THAT WAY. Every bar here is the scaffold cluster
        # bootstrap's 2.5/97.5 percentiles, and that distribution is skewed on the small panels.
        # Collapsing it to a symmetric +-1 value would hide the skew exactly where it is largest,
        # so yerr carries the two sides separately: [value - lo, hi - value].
        vals, lohi = [], []
        for enc_label, arm_key, _ in SERIES:
            ys, dn, up = [], [], []
            for probe in PROBES:
                r = g_all[(g_all.encoder == enc_label) & (g_all.probe == probe)]
                if not len(r):
                    ys.append(np.nan); dn.append(0.0); up.append(0.0); continue
                v, lo_, hi_ = (float(r.value.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0]))
                ys.append(v)
                dn.append(max(0.0, v - lo_))
                up.append(max(0.0, hi_ - v))
                lohi += [lo_, hi_]
            colour = ARMS[arm_key]["color"]
            _ls = _ls_for(enc_label)
            if _ls != "-":
                _ls_kinds.append((p, enc_label))
            ax.errorbar([0, 1], ys, yerr=[dn, up], color=colour, lw=STYLE["lw"], marker="o",
                        ls=_ls,
                        ms=4.4, mec="white", mew=0.8, elinewidth=1.0, capsize=3.0,
                        capthick=1.1, ecolor=colour, zorder=3)
            vals += [v for v in ys if np.isfinite(v)]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=FS["annot"])
        ax.set_xlim(-0.32, 1.32)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        if av is not None:
            lohi.append(av)
        lo, hi = min(lohi or vals), max(lohi or vals)
        pad = 0.22 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=ARMS[k]["color"], marker="o", ms=5.0, lw=1.4, label=lab)
               for _, k, lab in SERIES]
    handles.append(Line2D([], [], color=ARMS[ANCHOR_ARM]["color"], ls=":", lw=1.3,
                          label="XGBoost, ECFP4+desc"))
    # WIDTH FIRST: spend the page's width on the legend before its height (user 2026-08-19).
    # A legend row costs every figure below it on the page; a legend column costs nothing
    # until it runs past the text block, and these entries do not.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.068), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.0, frameon=False, labelcolor=INK)
    # Legend sits one text-height under the tick labels; see the SI b/d/e note.
    fig.tight_layout(rect=(0, 0.088, 1, 1), w_pad=0.35)
    save(fig, "SI_fig_a")
    plt.close(fig)

    print("\nSI Fig a — end2end minus frozen at full data (+ = end2end better):")
    for p in PANEL_ORDER:
        g = DF[DF.panel == p]
        if g.empty:
            print(f"   {p:<12} — no end2end run of a pretrained encoder")
            continue
        sign = 1 if g.higher_better.iloc[0] else -1
        for label, _, _ in SERIES:
            fr = g[(g.encoder == label) & (g.probe == "frozen")]
            ee = g[(g.encoder == label) & (g.probe == "end2end")]
            if not len(fr) or not len(ee):
                continue
            fr, ee = fr.iloc[0], ee.iloc[0]
            delta = sign * (float(ee.value) - float(fr.value))
            # DISJOINT INTERVALS, not |delta| > combined SD. The bars are bootstrap percentiles
            # now, and adding two percentile half-widths in quadrature is not a defined quantity.
            # Non-overlap is the statement the intervals actually support.
            flag = "*" if (fr.hi < ee.lo or ee.hi < fr.lo) else " "
            print(f"   {p:<12}{label:<20}{delta:>+10.4f}{flag}   ({fr.protocol})")
    print("   * = the frozen and end2end intervals are disjoint")


if __name__ == "__main__":
    main()

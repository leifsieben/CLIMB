"""SI Fig a — do you need to train the models end-to-end on the downstream data?

ONE script, ONE figure: figures_v2/SI_fig_a.png / .pdf

The same pretrained encoder used two ways at FULL downstream data: frozen (encoder fixed, probe
trained on the labels) versus end-to-end (whole network fine-tuned). Two encoders, `unsupervised`
and `supervised, dense`, on each canonical panel.

THE ANSWER IS MOSTLY YES, but it is not universal. End-to-end wins 9 of the 12 encoder x panel
cells and clears the combined SD in 5 of them. The largest gain by far is CBS with the unsupervised
encoder (+0.125 NEF1) — fine-tuning lifts the MLM encoder most of the way to the best frozen
supervised arm, though not past it. Then QM7 unsupervised (-12.0 RMSE) and Tox21 (+0.017 / +0.039).
All three exceptions involve the SUPERVISED encoder — CBS (-0.015), BACE (-0.013) and QM7
(-3.3 RMSE) — where freezing is as good or better. Read together: end-to-end training mostly buys
back what a weak pretraining objective failed to learn, and buys least where the frozen features
were already good.

So end-to-end fine-tuning is the better default, but the frozen probe is not far behind on several
panels, and it is the cheaper option by far (SI Fig c). SI Fig e shows how this trade depends on
how many labels you have: the frozen probe's advantage lives in the small-data regime.

Error bars are +-1 SD of that panel's replicate unit, and each panel's frozen and end2end numbers
come from the SAME wave, split and seed grid, so the within-panel comparison is like-for-like.

PROTOCOL WARNING — the protocol DIFFERS BETWEEN PANELS (MoleculeACE/Ames use the mainline wave;
CBS its 5 benchmark-provided folds; BACE/Tox21/QM7 the label-efficiency wave at its 100% fraction).
Compare frozen vs end2end WITHIN a panel; never compare a value in one panel against a value in
another. All 6 panels are populated as of 2026-08-18 — the CBS end-to-end runs of the two
pretrained encoders were the last gap and have now landed.

Data: figure_data/SI_fig_a/SI_fig_a_e2e_need.csv, built by scripts/build_SI_fig_a_table.py.

Run:  python3 scripts/build_SI_fig_a_table.py && python3 -m figures.SI_fig_a
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS, PANELS, PANEL_ORDER
from figures.sixpanel import ROOT

check_font()
INK = "#000000"

DF = pd.read_csv(ROOT / "figure_data" / "SI_fig_a" / "SI_fig_a_e2e_need.csv")

# A SLOPE plot, not bars (user request 2026-08-17): each encoder is ONE line joining its frozen
# value to its end2end value, so the only thing the reader has to judge is the DIRECTION and
# steepness of the change — which is the whole question — rather than comparing four bar heights.
# Colour = encoder family: blue = unsupervised, red = supervised, dense.
SERIES = [("unsupervised",      "unsup",     "unsupervised"),
          ("supervised, dense", "sup_dense", "supervised, dense")]
PROBES = ["frozen", "end2end"]
XTICKS = ["frozen", "end-to-end"]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 4.3))
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

        if g_all.empty:
            ax.text(0.5, 0.5, "no end2end run of a\npretrained encoder",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        vals, errs = [], []
        for enc_label, arm_key, _ in SERIES:
            ys, es = [], []
            for probe in PROBES:
                r = g_all[(g_all.encoder == enc_label) & (g_all.probe == probe)]
                ys.append(float(r.value.iloc[0]) if len(r) else np.nan)
                e = float(pd.to_numeric(r.sd, errors="coerce").iloc[0]) if len(r) else 0.0
                es.append(0.0 if not np.isfinite(e) else e)
            colour = ARMS[arm_key]["color"]
            ax.errorbar([0, 1], ys, yerr=es, color=colour, lw=STYLE["lw"], marker="o",
                        ms=5.4, mec="white", mew=0.8, elinewidth=1.0, capsize=2.2,
                        capthick=1.1, ecolor=colour, zorder=3)
            vals += [v for v in ys if np.isfinite(v)]
            errs += es

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=FS["annot"])
        ax.set_xlim(-0.32, 1.32)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        lo, hi = min(vals) - max(errs), max(vals) + max(errs)
        pad = 0.22 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if d["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Line2D([], [], color=ARMS[k]["color"], marker="o", ms=5.0, lw=1.4, label=lab)
               for _, k, lab in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=2,
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
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
            delta = sign * (float(ee.value.iloc[0]) - float(fr.value.iloc[0]))
            sd = np.hypot(pd.to_numeric(fr.sd, errors="coerce").iloc[0],
                          pd.to_numeric(ee.sd, errors="coerce").iloc[0])
            flag = "*" if np.isfinite(sd) and abs(delta) > sd else " "
            print(f"   {p:<12}{label:<20}{delta:>+10.4f}{flag}   ({g.protocol.iloc[0]})")
    print("   * = |delta| exceeds the combined SD")


if __name__ == "__main__":
    main()

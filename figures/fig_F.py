"""Fig F -- freeze or fine-tune? Where end2end overtakes the frozen probe.

ONE script, ONE figure: figures_v2/figF.png / .pdf

A pretrained encoder can be used two ways: freeze it and fit a probe, or fine-tune the whole
network end-to-end. The frozen probe has far fewer free parameters, so it should win when labels
are scarce; end2end has more capacity, so it should win once labels are plentiful. This figure
asks where the switch happens, using the SAME two encoders, the SAME label fractions, the SAME
hold-out split and the SAME seed grid on both sides -- only the probe strategy differs.

Plotted quantity: end2end MINUS frozen, signed so that POSITIVE always means end2end is better
(QM7 is RMSE, so its difference is negated). The dotted line at 0 is "the two strategies tie".
Bands are +-1 SD of the difference, sd = sqrt(sd_e2e^2 + sd_frozen^2) over the (subsample x head)
seed cells -- 9 per point at 5/10/25/50%, 3 at 100%.

What it actually shows -- report this honestly, it is NOT a clean crossover:
  Tox21 (7.8k, the largest panel) is the only task where the expected pattern resolves: end2end's
    advantage GROWS with training size, reaching +0.039 ROC-AUC for the supervised encoder at full
    data, well outside the seed noise.
  BACE (1.5k) is inside the noise at every fraction -- the sign flips between neighbouring points,
    so no crossover can be claimed.
  QM7 shows end2end ahead at EVERY fraction for the unsupervised encoder (by 9-21 kcal/mol), i.e.
    the opposite of the small-data expectation; the frozen unsupervised probe is unusually weak on
    QM7 (212.7 RMSE at full data), which is what that gap is measuring.
Conclusion for the text: fine-tuning end-to-end only reliably pays off on the largest task, and
the size of the effect is small next to the choice of pretraining recipe.

Scope: BACE, Tox21, QM7 -- the panels where a crossover can exist. MoleculeACE (<=3.7k per target)
and hERG (132 test molecules) are entirely small-data and CBS e2e is a separate fine-tuning path,
so Wave 3 deliberately did not run them. This figure is on the label-efficiency single hold-out
split, NOT the 5-fold CV of A2/B -- absolute values are not comparable across those figures.

Data: figure_data/figF/figF_crossover.csv, built by scripts/build_figF_table.py.

Run:  python3 scripts/build_figF_table.py && python3 -m figures.fig_F
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from figures.style import STYLE, FS, save, check_font
from figures.arms import FAMILY_COLORS

check_font()

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "figure_data" / "figF" / "figF_crossover.csv"

TASKS = [("BACE", "BACE  (1.5k, ROC-AUC)"),
         ("Tox21", "Tox21  (7.8k, ROC-AUC)"),
         ("QM7", "QM7  (6.8k, RMSE)")]
ENCODERS = [("unsup", "unsupervised", FAMILY_COLORS["unsup"]),
            ("sup", "supervised, dense", FAMILY_COLORS["sup"])]


def main():
    d = pd.read_csv(TABLE)
    fig, axes = plt.subplots(1, 3, figsize=(STYLE["col2"], 2.5), sharex=True)

    for ax, (task, subtitle) in zip(axes, TASKS):
        sign = -1 if d[d.task == task].direction.iloc[0] == "lower" else 1
        t = d[d.task == task]
        for key, label, colour in ENCODERS:
            fr = t[(t.encoder == key) & (t.probe == "frozen")].set_index("pct")
            ee = t[(t.encoder == key) & (t.probe == "end2end")].set_index("pct")
            pcts = sorted(set(fr.index) & set(ee.index))
            n = [fr.loc[p, "n_train"] for p in pcts]
            delta = np.array([sign * (ee.loc[p, "mean"] - fr.loc[p, "mean"]) for p in pcts])
            sd = np.array([np.hypot(pd.to_numeric(ee.loc[p, "sd"], errors="coerce"),
                                    pd.to_numeric(fr.loc[p, "sd"], errors="coerce"))
                           for p in pcts])
            sd = np.nan_to_num(sd)
            ax.fill_between(n, delta - sd, delta + sd, color=colour, alpha=0.16, lw=0, zorder=2)
            ax.plot(n, delta, "-o", color=colour, lw=STYLE["lw"],
                    ms=STYLE["marker_size"] * 0.55, label=label, zorder=3)

        ax.axhline(0, color=STYLE["ink"], lw=0.8, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("labelled training molecules")
        ax.set_title(subtitle, fontsize=FS["title"], fontweight="bold", color=STYLE["ink"], pad=4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(axis="y", ls=":", lw=0.5, color=STYLE["grid"])
        ax.set_axisbelow(True)

    axes[0].set_ylabel("end2end − frozen\n(+ = end2end better)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=FS["legend"],
                   handletextpad=0.5, borderpad=0.2, labelspacing=0.25)
    # a shaded "end2end ahead" half-plane reads instantly; only on the first panel to stay quiet
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.axhspan(0, hi, color=STYLE["faint"], zorder=0)
        ax.set_ylim(lo, hi)

    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.105, right=0.995, wspace=0.30)
    save(fig, "figF")
    plt.close(fig)

    print("\nFig F — end2end minus frozen (+ = end2end better), by labelled training size:")
    for task, _ in TASKS:
        sign = -1 if d[d.task == task].direction.iloc[0] == "lower" else 1
        t = d[d.task == task]
        for key, label, _ in ENCODERS:
            fr = t[(t.encoder == key) & (t.probe == "frozen")].set_index("pct")
            ee = t[(t.encoder == key) & (t.probe == "end2end")].set_index("pct")
            pcts = sorted(set(fr.index) & set(ee.index))
            cells = []
            for p in pcts:
                dl = sign * (ee.loc[p, "mean"] - fr.loc[p, "mean"])
                sd = np.hypot(pd.to_numeric(ee.loc[p, "sd"], errors="coerce"),
                              pd.to_numeric(fr.loc[p, "sd"], errors="coerce"))
                flag = "*" if np.isfinite(sd) and abs(dl) > sd else " "
                cells.append(f"{p:>4}%:{dl:+8.4f}{flag}")
            print(f"   {task:<6} {label:<18} " + " ".join(cells))
    print("   * = |difference| exceeds 1 SD of the difference")


if __name__ == "__main__":
    main()

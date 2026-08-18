"""Fig F — are CLIMB embeddings redundant to classical features?

ONE script, ONE figure: figures_v2/SI_Fig_d.png / .pdf  (+ figF.csv as the data record)

The test is concatenation. If the CLIMB embedding carries signal the classical features do not,
gluing it onto ECFP4+descriptors must beat ECFP4+descriptors alone. If it carries nothing new the
concatenation is at best flat — and, because the extra dimensions cost degrees of freedom, may be
slightly worse.

Four feature sets, same XGBoost head, same splits, same seeds:

  fp+desc       ECFP4 (2048 bits) + 217 RDKit descriptors   — the classical anchor
  CLM           the frozen CLIMB embedding alone
  desc+CLM      descriptors + CLIMB   (drops the fingerprint)
  fp+desc+CLM   everything            — the concatenation test

THE RESULT: concatenation helps on 1 of 6 tasks. On five — ESOL, QM7, BACE, Tox21, HIV —
`fp+desc+CLM` is WORSE than `fp+desc` alone, clearly so on the two regressions (ESOL 0.757 vs
0.730 RMSE; QM7 190.5 vs 187.5). The single exception is BBBP (+0.048 ROC-AUC, beyond its SD) —
and BBBP is exactly the dataset dropped from the paper's panel set for failing to discriminate:
its whole field spans 1.8% of ROC-AUC and an UNTRAINED random encoder ranks 7 of 16 on it
(notes/bbbp-anchor-verification-2026-08-16.md). A gain on the one benchmark we already decided
cannot separate models does not rescue the conclusion.

So on every benchmark that discriminates, the CLIMB embedding is redundant to the classical
featurization, and CLIMB alone is the weakest of the four feature sets on five of six tasks. This
is a negative result and is reported as one. It is also the honest frame for Fig A1, where
ECFP+desc ranks first overall: the transformer is not adding a missing view of the molecule.

PANEL SCOPE: the concatenation experiment was run on MoleculeNet tasks only, so of the canonical
six only BACE, Tox21 and QM7 are filled; MoleculeACE, CBS and hERG are drawn empty. ESOL, BBBP and
HIV were also run and are NOT shown here — they are outside the canonical panel set — but they are
in figF.csv and BBBP is the exception discussed above.

Error bars are +-1 SD across the seeds of that (task, feature set) cell.

Source: analysis/rigor/concat_redundancy.csv (git-tracked).

Run:  python3 -m figures.fig_F
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from figures.style import STYLE, FS, save, check_font
from figures.arms import PANELS, PANEL_ORDER, SHADES

check_font()
INK = "#000000"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "analysis" / "rigor" / "concat_redundancy.csv"
OUTDIR = ROOT / "figures_v2"

# canonical panel -> task name in the source (None = experiment never run there)
PANEL_TASK = {"MoleculeACE": None, "CBS": None, "BACE": "BACE",
              "Ames": None, "Tox21": "Tox21", "QM7": "QM7"}
PRIMARY = {"BACE": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
# every task in the source, for the CSV record (superset of the canonical panels)
ALL_TASKS = {"ESOL": "rmse", "QM7": "rmse", "BACE": "roc_auc", "BBBP": "roc_auc",
             "Tox21": "roc_auc", "HIV": "roc_auc"}
# classical anchor keeps the anchor amber; anything containing CLIMB moves into the unsup blues,
# darkening as more classical information is added back
FEATURES = [("fp+desc", "ECFP4 + descriptors (classical anchor)", SHADES["anchor"][0]),
            ("CLM", "CLIMB alone", SHADES["unsup"][2]),
            ("desc+CLM", "CLIMB + descriptors", SHADES["unsup"][1]),
            ("fp+desc+CLM", "CLIMB + descriptors + ECFP4", SHADES["unsup"][0])]
BASE, CONCAT = "fp+desc", "fp+desc+CLM"


def main():
    d = pd.read_csv(SRC)

    # ---- data record: every task the experiment covers, not just the canonical panels ----
    rows = []
    for task, metric in ALL_TASKS.items():
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        if BASE not in g.index or CONCAT not in g.index:
            continue
        sign = -1 if metric == "rmse" else 1
        delta = sign * (float(g.loc[CONCAT, "mean"]) - float(g.loc[BASE, "mean"]))
        sd = float(g.loc[CONCAT, "std"])
        row = dict(task=task, metric=metric,
                   in_canonical_panels=int(task in {v for v in PANEL_TASK.values() if v}))
        for f, _, _ in FEATURES:
            row[f] = round(float(g.loc[f, "mean"]), 4) if f in g.index else ""
            row[f + "_sd"] = round(float(g.loc[f, "std"]), 4) if f in g.index else ""
        row.update(delta_vs_fp_desc=round(delta, 4), concat_sd=round(sd, 4),
                   beats_sd="yes" if delta > sd else "no")
        rows.append(row)
    OUTDIR.mkdir(exist_ok=True)
    cols = ["task", "metric", "in_canonical_panels"] + \
           [c for f, _, _ in FEATURES for c in (f, f + "_sd")] + \
           ["delta_vs_fp_desc", "concat_sd", "beats_sd"]
    with open(OUTDIR / "figF.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- the figure: canonical six ----
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 5.1))
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        meta = PANELS[p]
        task = PANEL_TASK[p]
        arrow = "↑" if meta["higher_better"] else "↓"
        ax.set_title(f"{meta['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(meta["metric_short"], fontsize=FS["annot"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.5, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        if task is None:
            ax.text(0.5, 0.5, "concatenation test\nnot run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        metric = PRIMARY[task]
        g = d[(d.task == task) & (d.metric == metric)].set_index("features")
        x = np.arange(len(FEATURES))
        ys = [float(g.loc[f, "mean"]) if f in g.index else np.nan for f, _, _ in FEATURES]
        es = [float(g.loc[f, "std"]) if f in g.index else 0.0 for f, _, _ in FEATURES]
        cs = [c for _, _, c in FEATURES]
        ax.bar(x, ys, width=0.72, color=cs, edgecolor=INK, linewidth=0.8,
               yerr=es, error_kw=dict(elinewidth=1.0, capsize=2.2, capthick=1.1,
                                      ecolor=INK, zorder=6), zorder=3)
        # the classical anchor as a reference line makes "does adding CLIMB beat it?" readable
        ax.axhline(ys[0], color=SHADES["anchor"][0], ls=":", lw=1.1, zorder=2)
        ax.set_xticks(x)
        # The three CLIMB bars read as a cumulative build-up: CLIMB alone, then descriptors
        # added, then the fingerprint added. The classical anchor keeps its own name.
        ax.set_xticklabels(["ECFP\n+desc", "CLIMB", "+desc", "+fp"], fontsize=FS["annot"])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)
        lo = min(v - e for v, e in zip(ys, es) if np.isfinite(v))
        hi = max(v + e for v, e in zip(ys, es) if np.isfinite(v))
        pad = 0.30 * max(hi - lo, 1e-9)
        y0, y1 = lo - pad, hi + pad
        if meta["metric"] == "roc_auc":
            y1 = min(y1, 1.0)
        ax.set_ylim(y0, y1)

    handles = [Patch(facecolor=c, edgecolor=INK, lw=0.8, label=lab) for _, lab, c in FEATURES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=4,
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
               borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "figF")
    plt.close(fig)

    print("\nFig F — does concatenating CLIMB onto the classical features help?")
    print("  (delta signed so + = concatenation helped)\n")
    print(f"  {'task':<7}{'canon':<7}" + "".join(f"{f:>16}" for f, _, _ in FEATURES) +
          f"{'delta':>10}{'> SD?':>7}")
    for r in rows:
        line = f"  {r['task']:<7}{'yes' if r['in_canonical_panels'] else '—':<7}"
        for f, _, _ in FEATURES:
            line += f"{r[f]:>16.4f}"
        line += f"{r['delta_vs_fp_desc']:>+10.4f}{r['beats_sd']:>7}"
        print(line)
    helped = sum(r["beats_sd"] == "yes" for r in rows)
    print(f"\n  concatenation beat its own SD on {helped}/{len(rows)} tasks")
    print("  wrote figures_v2/SI_Fig_d.png/pdf + figF.csv")


if __name__ == "__main__":
    main()

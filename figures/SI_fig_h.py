"""SI Fig h — how much of a benchmark score is the SPLIT? Random vs scaffold, nothing else changed.

ONE script, ONE figure: figures_v2/SI_fig_h.png / .pdf

THE QUESTION (Leif 2026-08-29). Every headline number in this paper is a scaffold-split number,
which is the honest default: test scaffolds are unseen in training, so the score estimates
generalisation to new chemistry rather than interpolation within a congeneric series. A random
split does not do that — close analogues land on both sides — and it is what a great deal of
published work reports. So, holding the model, the data, the folds, the head and the seeds fixed
and changing ONE FLAG: how far apart are the two answers, and does the RANKING survive?

Read it left to right. Left = random 5-fold, right = scaffold 5-fold. The SLOPE is the split
penalty. A CROSSING is a ranking that exists only under one split — the thing a reader of the
literature cannot see, because papers report one split and not both.

FOUR ARMS, chosen so the contrast is between model FAMILIES rather than between budgets:
    CLIMB 100M supervised    skip_dense_100M_c124   the two halves of the 100M pair: same corpus,
    CLIMB 100M unsupervised  unsup_100M             same budget, objective the only difference
    ECFP4 + XGBoost          the anchor the CLMs are measured against
    ECFP4+desc + XGBoost     the stronger anchor, drawn as fig_B's reference line

BOTH ENDS ARE RUN FRESH IN THIS WAVE. Scaffold numbers for all four arms already existed and are
deliberately NOT reused: this repository has measured waves disagreeing by up to 8% on the same
model through the same code (Tox21 0.7356 against 0.7961). A slope whose left end came from this
wave and right end from another would draw the WAVE as much as the split — and the slope is the
entire message. Same code path, same 5 folds, same 3 head seeds, one flag different.

WHY AMES IS EMPTY, and it is structural rather than a cost we declined to pay. Polaris withholds
the test labels (scripts/chemeleon_suite_fetch_polaris.py: "test labels intentionally absent"), so
the evaluation set is fixed and hidden and cannot be re-partitioned at all. A random-vs-scaffold
comparison would have to happen inside the 5,821 labelled TRAINING molecules and would not be the
Ames number reported anywhere else in the paper. The panel is kept in place, labelled, rather than
silently reshaping the figure to the five tasks that have data — and PANEL_ORDER puts Ames last so
that costs the layout nothing.

WHISKERS are ±1 SD over the 5 folds, the same estimand at both ends of every line, so the whisker
describes the same quantity the slope is drawn between. MoleculeACE is the macro RMSE over its 30
targets and its whisker is the SD across those targets — a different unit from a fold SD, stated
here because the panel sits beside four that are not.

Data: figure_data/SI_fig_h/, built by scripts/si_fig_h_split_sensitivity.py
Run:  python3 scripts/si_fig_h_split_sensitivity.py && python3 -m figures.SI_fig_h
"""
from __future__ import annotations
import csv
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, row_ncol, LEGEND_BOX, mark_empty
from figures.arms import ARMS, PANELS, PANEL_ORDER

check_font()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figure_data" / "SI_fig_h"
INK = "#000000"

SCHEMES = [("random", "random"), ("scaffold", "scaffold")]
# arm dir token -> arms.py key for colour and label. One map, checked against the registry, after
# the same duplicate-lookup defect bit build_SI_fig_e_table and fig_A1 today.
LINES = {"ecfp": "ecfp", "ecfp_desc": "ecfp_desc",
         "unsup_100M": "unsup_100M", "sup_dense_100M": "sup_dense_100M"}
_bad = [v for v in LINES.values() if v not in ARMS]
assert not _bad, f"SI_fig_h.LINES maps to arm key(s) absent from arms.py: {_bad}"

# panel -> (dataset key in the eval output, metric). MoleculeACE is assembled from its 30 targets.
PANEL_SRC = {"BACE": ("BACE", "roc_auc"), "Tox21": ("Tox21", "roc_auc"),
             "QM7": ("QM7", "rmse"), "HIV": ("HIV", "nef1")}
AMES_WHY = "Polaris withholds\nthe test labels —\nno re-split possible"


def _folds(path: Path, dataset: str, metric: str):
    """The 5 per-fold values for one (cell, dataset, metric), or [].

    Reads the PLAIN metric rows, never `<metric>_cell`: eval_v2 averages the 3 head seeds'
    predictions before scoring, and that ensembled score is the intended point estimate. Averaging
    `_cell` rows instead is a different, strictly worse estimator — the bug that understated every
    BACE/Tox21 arm by 0.5-1% AUC in 2026-08.
    """
    if not path.exists():
        return []
    return [float(r["main_value"]) for r in csv.DictReader(path.open())
            if r["dataset"] == dataset and r["main_metric"] == metric
            and r["head_seed"] not in ("MEAN", "STD") and r["main_value"] not in ("", "nan")]


def cell(arm: str, scheme: str, panel: str):
    """(value, sd) for one point, or (nan, nan)."""
    if panel == "MoleculeACE":
        # macro RMSE over the 30 targets; the spread is ACROSS TARGETS, not across folds.
        base = SRC / f"{arm}__{scheme}__mace"
        per = []
        for d in sorted(base.glob("*")):
            v = _folds(d / "moleculenet_summary.csv", d.name, "rmse")
            if v:
                per.append(st.mean(v))
        if not per:
            return float("nan"), float("nan")
        return st.mean(per), (st.stdev(per) if len(per) > 1 else float("nan"))
    if panel not in PANEL_SRC:
        return float("nan"), float("nan")
    ds, metric = PANEL_SRC[panel]
    v = _folds(SRC / f"{arm}__{scheme}" / "moleculenet_summary.csv", ds, metric)
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan"))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.9))
    x = np.array([0.0, 1.0])
    drawn_any = False
    for ax, panel in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[panel]
        arrow = "↑" if d["higher_better"] else "↓"
        ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        vals = {a: [cell(a, s, panel) for s, _ in SCHEMES] for a in LINES}
        if not any(np.isfinite(v) for pair in vals.values() for v, _ in pair):
            # Kept in place and LABELLED. An unexplained empty panel and a panel whose data is
            # structurally unobtainable look identical, and only one of them is a finding.
            # mark_empty() only DECLARES the intent for check_no_empty_panels; it draws nothing.
            # Without the text and the stripped axes below the panel renders as a bare 0-1 grid,
            # which reads as a plotting bug rather than as a statement about the benchmark.
            mark_empty(ax, AMES_WHY)
            ax.text(0.5, 0.5, AMES_WHY, transform=ax.transAxes, ha="center", va="center",
                    fontsize=FS["annot"], color=INK)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{d['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                         color=INK, pad=4)
            continue
        drawn_any = True
        for arm, key in LINES.items():
            y = np.array([v for v, _ in vals[arm]], dtype=float)
            e = np.array([s for _, s in vals[arm]], dtype=float)
            if not np.isfinite(y).all():
                continue
            c = ARMS[key]["color"]
            ax.plot(x, y, color=c, lw=STYLE["lw"], marker="o", ms=4.4, mec="white", mew=0.6,
                    zorder=3)
            # NaN sd passes through: matplotlib omits it. A missing spread must not draw as a
            # zero-length whisker, which is a claim of perfect precision.
            ax.errorbar(x, y, yerr=e, fmt="none", ecolor=c, elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=2)
        ax.set_xlim(-0.30, 1.30)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random", "scaffold"], fontsize=FS["annot"])
        ax.set_ylabel(d["metric_short"], fontsize=FS["annot"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=FS["annot"] - 1)
    assert drawn_any, "SI_fig_h: every panel is empty -- run scripts/si_fig_h_split_sensitivity.py"

    h = [Line2D([], [], color=ARMS[k]["color"], marker="o", ms=4.4, lw=1.4,
                label=ARMS[k]["label"] if k.startswith("ecfp") else
                f"{ARMS[k]['label']} 100M") for k in LINES.values()]
    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=0.5)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.085),
               ncol=row_ncol(h, rows=1), fontsize=FS["legend"], handletextpad=0.5,
               columnspacing=1.3, borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    save(fig, "SI_fig_h")
    plt.close(fig)
    report()


def report():
    print("\nSI Fig h — split penalty (random - scaffold), same code path both ends\n")
    print(f"   {'panel':13s}{'arm':18s}{'random':>10s}{'scaffold':>10s}{'penalty':>10s}")
    for panel in PANEL_ORDER:
        for arm in LINES:
            (r, _), (s, _) = cell(arm, "random", panel), cell(arm, "scaffold", panel)
            if not (np.isfinite(r) and np.isfinite(s)):
                continue
            print(f"   {panel:13s}{arm:18s}{r:10.4f}{s:10.4f}{r - s:+10.4f}")


if __name__ == "__main__":
    main()

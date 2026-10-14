from __future__ import annotations
from pathlib import Path
import csv
import statistics as st

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol, LEGEND_BOX
from figures.arms import ARMS, series_label, PANELS, PANEL_ORDER

check_font()
ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
INK = "#000000"


HEADS = ["mlp", "xgb"]
XTICKS = ["MLP", "XGBoost"]


SERIES = [("ecfp_desc", "fp_desc_anchor", "xgb"),
          ("unsup", "unsup_8M", "mlp"),
          ("sup_dense", "skip_dense_8M", "mlp")]


METRIC = {"MoleculeACE": "rmse", "HIV": "nef1", "BACE": "roc_auc",
          "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}


def _mainline():

    out = {}
    f = FD / "six_panel" / "mainline_8M.csv"
    if not f.exists():
        return out
    for r in csv.DictReader(f.open()):
        try:
            out[(r["arm"], r["panel"])] = float(r["value"])
        except (ValueError, KeyError):
            pass
    return out


def _stem(arm, panel):

    src = ARMS[arm]["src"]
    key = "mace" if panel in ("MoleculeACE", "Ames") else "mol"
    v = src.get(key) or src.get("mol")
    return v[0] if isinstance(v, (list, tuple)) else v


def _head_cell(arm, head, panel):


    task = panel
    if panel in ("MoleculeACE", "Ames"):
        track = "moleculeace" if panel == "MoleculeACE" else "polaris"
        d = FD / "chemeleon_suite" / track / f"{_stem(arm, panel)}__{head}"
        f = d / ("results.csv" if panel == "MoleculeACE" else "polaris_scores.csv")
        if not f.exists():
            return None
        vals = []
        for r in csv.DictReader(f.open()):
            if panel == "MoleculeACE":
                if r.get("subset") == "overall" and r.get("metric") == "rmse":
                    vals.append(float(r["value"]))
            elif (r.get("task") == PANELS[panel]["polaris_task"]
                  and r.get("metric") == PANELS[panel]["metric"]
                  and r.get("value") not in ("", "nan")):
                vals.append(float(r["value"]))
        return st.mean(vals) if vals else None

    f = (FD / "climb_v2_phase2" / f"{_stem(arm, panel)}__{head}" / "moleculenet_cv"
         / "moleculenet_summary.csv")
    if not f.exists():
        return None
    m = METRIC[panel]
    vals = [float(r["main_value"]) for r in csv.DictReader(f.open())
            if r["dataset"] == task and r["main_metric"] == m
            and r["head_seed"] not in ("MEAN", "STD") and r["main_value"] not in ("", "nan")]
    return st.mean(vals) if vals else None


def compute():
    """{panel: {arm: [value_per_head in HEADS order]}}, with None for anything not run."""
    main = _mainline()
    out = {}
    for p in PANEL_ORDER:
        cells = {}
        for arm, tag, main_head in SERIES:
            pair = {}
            pair[main_head] = main.get((arm, p))
            other = "mlp" if main_head == "xgb" else "xgb"
            pair[other] = _head_cell(arm, other, p)
            cells[arm] = [pair[h] for h in HEADS]
        out[p] = cells
    return out


def main():
    R = compute()
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"] * 1.035, 3.55))
    n_missing = 0
    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        meta = PANELS[p]
        arrow = "↑" if meta["higher_better"] else "↓"
        ax.set_title(f"{meta['label']} {arrow}", fontsize=FS["title"], fontweight="bold",
                     color=INK, pad=4)
        ax.set_ylabel(meta["metric_short"], fontsize=FS["annot"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        drawn, vals = 0, []
        for arm, _, _ in SERIES:
            ys = R[p][arm]
            if ys[0] is None or ys[1] is None:
                continue
            ax.plot([0, 1], ys, color=ARMS[arm]["color"], lw=STYLE["lw"], marker="o",
                    ms=5.0, mec="white", mew=0.8, zorder=3)
            drawn += 1
            vals += ys
        if not drawn:


            ax.text(0.5, 0.5, "head comparison\nnot run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS["annot"], color=INK)
            ax.set_xticks([]); ax.set_yticks([])
            mark_empty(ax, f"{p}: head comparison not run")
            n_missing += 1
            continue

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICKS, fontsize=FS["annot"])
        ax.set_xlim(-0.30, 1.30)
        lo, hi = min(vals), max(vals)
        pad = 0.22 * max(hi - lo, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)


    handles = [Line2D([], [], color=ARMS[a]["color"], marker="o", ms=5.0, lw=1.4,
                      label=series_label(a)) for a, _, _ in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=0.35)
    save(fig, "figure_F1")
    plt.close(fig)


    lo, hi = (XTICKS[0], XTICKS[1])
    print(f"\nSI Fig f — same embedding, two probe heads ({lo} -> {hi}):\n")
    print(f"   {'panel':<13}{'representation':<22}{lo:>10}{hi:>10}   delta")
    for p in PANEL_ORDER:
        for arm, _, _ in SERIES:
            a0, a1 = R[p][arm]
            if a0 is None or a1 is None:
                miss = lo if a0 is None else hi
                print(f"   {p:<13}{series_label(arm)[:20]:<22}{'—':>10}{'—':>10}   {miss} not run")
                continue
            print(f"   {p:<13}{series_label(arm)[:20]:<22}{a0:>10.4f}{a1:>10.4f}   {a1 - a0:>+8.4f}")
    if n_missing:
        print(f"\n   {n_missing} of {len(PANEL_ORDER)} panels have NO head-comparison run yet "
              f"(scripts/head_comparison_run.sh).")


if __name__ == "__main__":
    main()

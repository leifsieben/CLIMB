from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figures.sixpanel import (suite_run_mean, suite_wave_mean, canonical_value,
                              canonical_lift, crosswave_safe, report_crosswave,
                              joint_molnet_subdirs)
from figures.style import STYLE, FS, check_font, LEGEND_BOX
from figures.arms import ARMS, SHADES, TASK_COLORS, LIFT_YLABEL

check_font()

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "figure_data" / "climb_v2_ablation_dedup"
PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"
SIMP = ROOT / "figure_data" / "_tanimoto" / "family_task_similarity.csv"


ARM2FAM = {"seq_pcba": ["PCBA"],
           "seq_l1000": ["L1000_MCF7", "L1000_VCAP"],
           "seq_pcqm": ["PCQM"],
           "seq_sparse_all": ["PCBA", "L1000_MCF7", "L1000_VCAP"]}
ARM_LABEL = {"seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}

TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]
LOWER_BETTER = {"MoleculeACE", "QM7"}


MOLNET_TASKS = ["BACE", "Tox21", "QM7", "HIV"]

FLOOR_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN_ABL = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN_PH2 = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["random_encoder"]["label"]


def _suite_mean(run_dir, task):


    return suite_run_mean(run_dir, task)


def _suite_mean_raw(run_dir, task):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    d = json.load(open(p))
    v = d.get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def _wave_mean(wave, runs, task):

    return suite_wave_mean(wave, runs, task)


def _lift(arm_val, task, floor_val):
    return canonical_lift(arm_val, floor_val, task)


def _joint(task, arm):

    return joint_molnet_subdirs(task, [(ABL.name, [arm]), (ABL.name, FLOOR_RUNS)])


def _arm_value(arm, task):

    return canonical_value(task, dict(mace=arm, mol=arm), molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, arm))


def _floor_value(task, arm):

    return canonical_value(task, ARMS["random_encoder"]["src"], molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, arm))


def compute():
    """All C2 numbers, once. Used by Figure 3."""


    tasks, dropped = crosswave_safe(TASKS, ABL, PHASE2, FROZEN_ABL, MOLNET_TASKS)
    report_crosswave(dropped, FLOOR_LABEL)
    if not dropped:
        print(f"   frozen floors agree across both waves on all {len(MOLNET_TASKS)} MolNet tasks "
              f"(informational: the {FLOOR_LABEL} floor is now read IN-WAVE, nothing is borrowed)")

    sim = pd.read_csv(SIMP)
    pts = []
    for arm, fams in ARM2FAM.items():
        for t in tasks:
            av = _arm_value(arm, t)
            fv = _floor_value(t, arm)
            l = _lift(av, t, fv)
            m = sim[(sim.task == t) & (sim.family.isin(fams))].mean_max_tanimoto.mean()
            if np.isfinite(l) and np.isfinite(m):
                pts.append((m, l, t, arm))
    if len(pts) < 4:
        raise RuntimeError(f"only {len(pts)} arm x task cells -- cannot run the similarity test")

    X = np.array([p[0] for p in pts])
    Y = np.array([p[1] for p in pts])
    TK = [p[2] for p in pts]
    AM = [p[3] for p in pts]

    from scipy import stats as _st
    r = float(np.corrcoef(X, Y)[0, 1])
    rho, p = _st.spearmanr(X, Y)
    b, a = np.polyfit(X, Y, 1)
    return dict(pts=pts, X=X, Y=Y, TK=TK, AM=AM, r=r, rho=rho, p=p, a=a, b=b)


def draw(ax, data, tag=None, compact=False):
    """Draw the C2 scatter onto an existing axes (standalone or assembled context)."""
    pts, X, Y, TK, AM = data["pts"], data["X"], data["Y"], data["TK"], data["AM"]
    r, rho, p, a, b = data["r"], data["rho"], data["p"], data["a"], data["b"]


    for t in [t for t in TASKS if t in set(TK)]:
        sel = [k for k in range(len(pts)) if TK[k] == t]
        if sel:
            ax.scatter(X[sel], Y[sel], s=34, color=TASK_COLORS[t], marker="o",
                       edgecolor="white", lw=0.5, zorder=3)
    xs = np.linspace(X.min(), X.max(), 50)
    ax.plot(xs, a + b * xs, color=SHADES["random"][0], lw=STYLE["lw_thin"], ls=(0, (4, 2)),
            zorder=2)
    ax.axhline(0, color=SHADES["e2e"][0], lw=STYLE["lw_thin"], zorder=1)


    ax.set_xlabel("Average max TanSim pretrain \u2194 downstream" if compact else
                  "Average max ECFP4 TanSim: pretrain \u2194 downstream   (right = more similar)")
    ax.set_ylabel(LIFT_YLABEL)
    ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)

    task_handles = [Line2D([], [], color=TASK_COLORS[t], marker="o", ls="none", ms=5,
                           mec="white", mew=0.5, label=t) for t in TASKS if t in set(TK)]


    lo, hi = ax.get_ylim()
    span = hi - lo
    ax0, ax1 = X.min(), X.max()


    BW, BH = (0.62, 0.28) if compact else (0.52, 0.30)
    def CORNER_TESTS(ay0, ay1):
        xl = ax0 + BW * (ax1 - ax0); xr = ax1 - BW * (ax1 - ax0)
        yb = ay0 + BH * (ay1 - ay0); yt = ay1 - BH * (ay1 - ay0)
        return [("lower left",  lambda x, y: x <= xl and y <= yb),
                ("lower right", lambda x, y: x >= xr and y <= yb),
                ("upper left",  lambda x, y: x <= xl and y >= yt),
                ("upper right", lambda x, y: x >= xr and y >= yt)]


    for pad in (0.26, 0.34, 0.42, 0.50, 0.60):
        ax.set_ylim(lo - pad * span, hi + 0.06 * span)
        ay0, ay1 = ax.get_ylim()
        occupancy = [(n, [(x, y) for x, y in zip(X, Y) if f(x, y)]) for n, f in CORNER_TESTS(ay0, ay1)]
        empty = [n for n, pts in occupancy if not pts]
        if empty:
            break
    assert empty, ("fig_C2: every legend corner covers data even at 0.60 padding - "
                   + "; ".join(f"{n} {len(q)}" for n, q in occupancy)
                   + ". Split the legend rather than covering points.")
    loc = empty[0]


    kw = dict(handles=task_handles, loc=loc, **LEGEND_BOX)
    if compact:
        ax.legend(ncol=3, fontsize=FS["legend"] - 0.5, handletextpad=0.25, borderaxespad=0.35,
                  borderpad=0.3, labelspacing=0.22, columnspacing=0.6, handlelength=0.9, **kw)
    else:
        ax.legend(ncol=2, fontsize=FS["legend"], handletextpad=0.3, borderaxespad=0.4,
                  borderpad=0.45, labelspacing=0.3, **kw)
    print(f"   C2 legend -> {loc} at {pad:.2f} bottom padding ("
          + ", ".join(f"{n}:{len(q)}" for n, q in occupancy) + " points per corner)")


    ax.set_title("Supervised pretrain" if compact else
                 "Supervised pretraining: transfer vs chemical similarity",
                 loc="left" if compact else "center",
                 fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tag:
        ax.text(*((-0.09, 1.05) if compact else (-0.14, 1.02)), tag, transform=ax.transAxes,
                fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

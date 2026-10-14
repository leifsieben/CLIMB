from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.sixpanel import (suite_run_mean, suite_wave_mean, canonical_value,
                              canonical_lift, crosswave_safe, report_crosswave,
                              joint_molnet_subdirs)
from figures.style import STYLE, FS, check_font, LEGEND_BOX
from figures.arms import ARMS, SHADES, LIFT_YLABEL

check_font()

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "figure_data" / "climb_v2_ablation_dedup"
PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"


FAMILIES = ["seq_mtr", "seq_dense_plus_sparse", "seq_pcba", "seq_l1000", "seq_pcqm",
            "seq_sparse_all"]
FAM_LABEL = {"seq_mtr": "desc", "seq_dense_plus_sparse": "desc+sparse",
             "seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}
FAM_SHORT = {"seq_mtr": "desc (MTR)", "seq_dense_plus_sparse": "desc+sparse",
             "seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}


FAM_COL = {"seq_mtr": SHADES["u2s"][0], "seq_dense_plus_sparse": SHADES["u2s"][1],
           "seq_sparse_all": SHADES["u2s"][2], "seq_pcba": SHADES["u2s"][3],
           "seq_l1000": SHADES["u2s"][3], "seq_pcqm": SHADES["u2s"][4]}
FAM_MARKER = {"seq_mtr": "o", "seq_dense_plus_sparse": "s", "seq_pcba": "D",
              "seq_l1000": "v", "seq_pcqm": "^", "seq_sparse_all": "P"}
SLOPE_FAMS = ["seq_mtr", "seq_sparse_all"]

TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]


TASK_GROUP = {"QM7": "property",
              "MoleculeACE": "bioassay", "HIV": "bioassay", "BACE": "bioassay",
              "Ames": "bioassay", "Tox21": "bioassay"}
GROUPS = ["property", "bioassay"]
GROUP_LABEL = {"property": "property\nregression", "bioassay": "bioassay\nclassification"}
GROUP_MEMBERS = "descriptor-like: QM7          bioassay: MoleculeACE, CBS, BACE, Ames"
LOWER_BETTER = {"MoleculeACE", "QM7"}

MOLNET_TASKS = ["BACE", "Tox21", "QM7", "HIV"]

FLOOR_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["random_encoder"]["label"]


BASE_SRC = dict(mace="unsup_2M", mol=["unsup_2M"])
ANCHOR_SRC = ARMS["ecfp"]["src"]
BASE_LABEL = "unsupervised"
ANCHOR_LABEL = ARMS["ecfp"]["label"] + " (XGBoost)"


def _suite_mean(run_dir, task):


    return suite_run_mean(run_dir, task)


def _suite_mean_raw(run_dir, task):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    v = json.load(open(p)).get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def _wave_mean(wave, runs, task):
    return suite_wave_mean(wave, runs, task)


def _wave_mean_raw(wave, runs, task):
    vs = [v for v in (_suite_mean(wave / r, task) for r in runs) if np.isfinite(v)]
    return float(np.mean(vs)) if vs else np.nan


def _lift(arm_val, task, floor_val):
    return canonical_lift(arm_val, floor_val, task)


def _joint(task, run):
    """The one MolNet subdir BOTH trees can supply — see figures.sixpanel.joint_molnet_subdirs."""
    return joint_molnet_subdirs(task, [(ABL.name, [run]), (ABL.name, FLOOR_RUNS)])


def _arm_value(run, task):

    return canonical_value(task, dict(mace=run, mol=run), molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, run))


def _floor_value(task, run):

    return canonical_value(task, ARMS["random_encoder"]["src"], molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, run))


def compute():
    """All D numbers, once. Used by Figure 3."""

    tasks, dropped = crosswave_safe(TASKS, ABL, PHASE2, FROZEN, MOLNET_TASKS)
    report_crosswave(dropped, FLOOR_LABEL)


    H = pd.DataFrame(index=FAMILIES, columns=tasks, dtype=float)
    for fam in FAMILIES:
        for t in tasks:
            H.loc[fam, t] = _lift(_arm_value(fam, t), t, _floor_value(t, fam))


    base_row = {t: _lift(canonical_value(t, BASE_SRC, molnet_subdirs=_joint(t, BASE_SRC["mol"][0])),
                         t, _floor_value(t, BASE_SRC["mol"][0])) for t in tasks}
    anchor_row = {t: _lift(canonical_value(t, ANCHOR_SRC,
                                           molnet_subdirs=_joint(t, ANCHOR_SRC["mol"][0])),
                           t, _floor_value(t, ANCHOR_SRC["mol"][0])) for t in tasks}
    bar_rows = ([(BASE_LABEL, float(np.nanmean(list(base_row.values()))), ARMS["unsup"]["color"])]
                + [(FAM_LABEL[f], float(np.nanmean([H.loc[f, t] for t in tasks])), FAM_COL[f])
                   for f in FAMILIES]
                + [(ANCHOR_LABEL, float(np.nanmean(list(anchor_row.values()))),
                    ARMS["ecfp"]["color"])])
    return dict(H=H, bar_rows=bar_rows, tasks=tasks, dropped=dropped)


def draw(axB, axM, axS, data, tags=("a", "b", "c"), compact=False):

    H, bar_rows = data["H"], data["bar_rows"]


    vals = [r[1] for r in bar_rows]
    xmax = max(abs(min(vals)), abs(max(vals))) * 1.30 + 0.5
    axB.barh(range(len(bar_rows)), vals, color=[r[2] for r in bar_rows],
             height=0.62, edgecolor="white", lw=0.4)
    for i, v in enumerate(vals):
        axB.text(v + (0.12 if v >= 0 else -0.12), i, f"{v:+.1f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=FS["annot"])
    axB.axvline(0, color=SHADES["random"][0], lw=0.8)
    axB.set_yticks(range(len(bar_rows)))
    axB.set_yticklabels([r[0] for r in bar_rows], fontsize=FS["annot"])
    axB.invert_yaxis()
    axB.set_xlim(-xmax, xmax)
    axB.set_xlabel(f"Mean lift over {FLOOR_LABEL} (%)")
    axB.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"])
    axB.set_axisbelow(True)
    axB.set_title("Lift by SFT label type",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[0]:
        axB.text(*((-0.18, 1.05) if compact else (-0.80, 1.03)), tags[0], transform=axB.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")


    vmax = float(np.nanmax(np.abs(H.values)))
    vmax = max(20.0, np.ceil(vmax / 10) * 10)
    norm = mpl.colors.SymLogNorm(linthresh=5, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    im = axM.imshow(H.values, cmap="PuOr_r", norm=norm, aspect="auto")
    axM.set_xticks(range(len(H.columns)))
    axM.set_xticklabels(list(H.columns) if compact else
                        [f"{t}\n[{TASK_GROUP[t][:4]}.]" for t in H.columns],
                        fontsize=FS["annot"], rotation=30, ha="right")
    axM.set_yticks(range(len(FAMILIES)))
    axM.set_yticklabels([FAM_SHORT[f] for f in FAMILIES], fontsize=FS["annot"])
    axM.grid(False)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            if np.isfinite(v):


                lum = np.dot(mpl.colormaps["PuOr_r"](norm(v))[:3], (0.2126, 0.7152, 0.0722))
                axM.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=FS["annot"],
                         color="white" if lum < 0.5 else STYLE["ink"])


    cbt = [t for t in (-vmax, -5, 0, 5, vmax) if abs(t) <= vmax]


    cb = axM.figure.colorbar(im, ax=axM, orientation="horizontal", location="bottom",
                             fraction=0.055, pad=0.20, aspect=32, ticks=cbt)
    cb.ax.xaxis.set_major_formatter(ticker.FixedFormatter(
        [f"{t:+.0f}".replace("+0", "0") for t in cbt]))

    cb.set_label(LIFT_YLABEL if compact else LIFT_YLABEL + ", symlog", fontsize=FS["legend"])
    cb.ax.tick_params(labelsize=FS["annot"])


    axM.set_title("Pretrain task \u2192 Downstream task", loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[1]:
        axM.text(*((-0.14, 1.05) if compact else (-0.30, 1.03)), tags[1], transform=axM.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")


    for fam in SLOPE_FAMS:
        per_task = {g: [H.loc[fam, t] for t in H.columns if TASK_GROUP[t] == g] for g in GROUPS}
        means = [float(np.nanmean(per_task[g])) for g in GROUPS]
        axS.plot([0, 1], means, color=FAM_COL[fam], marker=FAM_MARKER[fam], ms=5,
                 mec="white", mew=0.6, lw=STYLE["lw"], zorder=3)
        for gi, g in enumerate(GROUPS):
            xs = np.full(len(per_task[g]), gi) + np.linspace(-0.10, 0.10, len(per_task[g]))
            axS.scatter(xs, per_task[g], s=16, color=FAM_COL[fam], marker=FAM_MARKER[fam],
                        alpha=0.55, zorder=2, edgecolor="none")
    handles = [Line2D([], [], color=FAM_COL[f], marker=FAM_MARKER[f], ls="-", ms=5,
                      lw=STYLE["lw"], mec="white", mew=0.6, label=FAM_SHORT[f])
               for f in SLOPE_FAMS]


    axS.legend(handles=handles, loc="lower left", **LEGEND_BOX, fontsize=FS["legend"],
               handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)
    axS.axhline(0, color=SHADES["random"][0], lw=0.6, zorder=1)
    axS.set_xlim(-0.38, 1.38)
    axS.set_xticks([0, 1])
    if compact:


        def _members(g):
            ms = [t for t in H.columns if TASK_GROUP[t] == g]
            return ", ".join(ms) if len(ms) < 3 else ", ".join(ms[:2]) + ",\n" + ", ".join(ms[2:])
        axS.set_xticklabels([f"property\n({_members('property')})",
                             f"bioassay\n({_members('bioassay')})"],
                            fontsize=FS["annot"])
    else:
        axS.set_xticklabels([GROUP_LABEL[g] for g in GROUPS], fontsize=FS["annot"])
        axS.set_xlabel(GROUP_MEMBERS, fontsize=FS["annot"])

    axS.set_ylabel(LIFT_YLABEL)
    axS.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    axS.set_axisbelow(True)
    axS.set_title("Descriptor-like task mapping",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[2]:
        axS.text(*((-0.12, 1.05) if compact else (-0.055, 1.03)), tags[2], transform=axS.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

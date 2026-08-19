"""Fig C2 -- does molecular similarity (SFT data <-> eval task) explain transfer of SUPERVISED
pretraining?  The H10 test, on the deduped ablation wave.

ONE script, ONE figure: figures_v2/figC2.png / .pdf

What it shows
-------------
Every point is one (SFT family, eval task) pair (24 = 4 families x 6 tasks -- the same cells as
the old ipynb H10 panel), coloured by eval task only: x = mean max ECFP4 Tanimoto similarity between
the family's pretraining molecules and the task's molecules (figure_data/_tanimoto/
family_task_similarity.csv; families are sampled, so similarities are lower bounds), y = the
arm's lift over the honest floor ("no pretrain, end2end" -- the random-init encoder fine-tuned
on the task; beating a frozen random encoder is close to automatic, so it is not the comparator).

Arms are the molecule-set SFT families of the deduped ablation wave (figure_data/
climb_v2_ablation_dedup): PCBA, L1000 (MCF7+VCAP pooled), PCQM, and sparse_all (PCBA+L1000).
All are unsup->sup warm-started from ONE shared 2M-FP MLM base -- the wave has no random-init
single-family arms, so "supervised pretraining" here always sits on top of that shared MLM base.
The dense descriptor arms (seq_mtr, seq_dense_plus_sparse) have NO family molecule set to measure
similarity against, so they have no x-coordinate and are absent by construction (they belong to
the task-similarity figure D, not here).

The deduped wave drops 34,301 eval molecules from SFT via a blocklist, so no arm trains on
eval-test molecules. All arms and the ablation-wave floor are 5-fold CV means (suite_summary.json
<task>_MEAN). The end2end floor lives in climb_v2_phase2; borrowing it cross-wave is safe only
while the two waves' FROZEN random-encoder floors agree (same three inits, both CV-scored) --
checked at runtime and printed, never assumed.

Eval tasks are the MoleculeNet tasks covered by the similarity table (CBS / MoleculeACE / hERG
have no family-similarity measurements). Lipophilicity drops out: the phase2 end2end floor runs
were never scored on it, so there is no honest floor to lift against (n=24 = 4 families x 6
tasks). seq_sparse_all's x pools its three families.

Run:  python3 -m figures.fig_C2

PANEL SET — MIGRATED to the canonical six on 2026-08-18. n=24 (4 families x 6 tasks), the same
design as the pre-canonical version. MoleculeACE / CBS / Ames values come from the shared benchmark trees
(chemeleon_suite/, cbs_benchmark/), resolved by figures.sixpanel.canonical_value, which delegates
to scripts/six_panel_aggregate.py so this figure's definition of a panel value is byte-identical to
every other figure's.

TOX21 IS DROPPED AGAIN as of 2026-08-18 (final state), and this time it is a hard data gap, not a
stale file. The ablation wave's per-molecule Tox21 dumps are PRE-fix: 93,876 rows rather than the
masked 77,864, so nothing in that tree can produce the corrected number and it cannot be re-scored
from disk at all -- it needs a re-eval against the checkpoints. Phase 2 IS corrected, so the two
trees are not comparable and the panel is withheld. Note this is deliberately NOT treated like the
QM7 unit split: falling back to the stale copy on both sides would give a lift of two wrong numbers,
internally consistent and still wrong, because the missing-label correction moves arms by
+0.015...+0.032 non-uniformly rather than rescaling them. See figures.sixpanel.CORRECTION_TASKS.

TWO GUARDS RUN HERE, and both caught a live fault on 2026-08-18. Neither is decorative.

1. CROSS-WAVE FLOOR AGREEMENT (figures.sixpanel.crosswave_safe). The lift borrows phase 2's end2end
   floor because the ablation wave has none of its own, which is valid only while both waves
   evaluate a task the same way. The test is the frozen random-encoder floor, which both waves
   have. It fired on Tox21: the same checkpoints read 0.7710 in the local ablation tree against
   0.7526 in phase 2 (+2.4%), an offset worth 34-72% of every seq_* family's Tox21 lift. Cause: a
   STALE LOCAL COPY, not a protocol difference — the ablation wave was first scored 2026-07-21, two
   weeks before commit 79a0dfb fixed Tox21's missing-label handling (DeepChem encodes missing
   multi-assay labels as y=0,w=0 rather than NaN, so ~16k missing entries were scored as true
   inactives and inflated ROC-AUC). S3 had the corrected re-score; the pre-fix file sat on disk.
   That also explains why BACE matched byte-for-byte: BACE is single-label, so the bug cannot touch
   it, and Tox21 is the only multi-label task in the six. After syncing, the waves agree to 0.000%
   and Tox21 is back in the figure.

2. JOINT UNIT RESOLUTION (figures.sixpanel.joint_molnet_subdirs). A lift is scale-invariant only
   when arm and floor are in the SAME convention. The phase-2 floor has a native QM7 re-eval and
   the ablation seq_* arms do not, so resolving each side's preferred subdir independently produced
   a +99.6% QM7 lift out of nothing. Both sides are now read from the one convention BOTH trees can
   supply. Nothing had to be re-run — the z-scored comparison is the same number.

The ESOL/Lipophilicity z-scored-vs-native unit defect that contaminated the pre-canonical version
is now MOOT: the canonical six contains neither task.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.sixpanel import (suite_run_mean, suite_wave_mean, canonical_value,
                              canonical_lift, crosswave_safe, report_crosswave,
                              joint_molnet_subdirs)
from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, SHADES, TASK_COLORS, LIFT_YLABEL

check_font()

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "figure_data" / "climb_v2_ablation_dedup"
PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"
SIMP = ROOT / "figure_data" / "_tanimoto" / "family_task_similarity.csv"

# SFT family arms -> the families whose molecules define their x-coordinate
ARM2FAM = {"seq_pcba": ["PCBA"],
           "seq_l1000": ["L1000_MCF7", "L1000_VCAP"],
           "seq_pcqm": ["PCQM"],
           "seq_sparse_all": ["PCBA", "L1000_MCF7", "L1000_VCAP"]}
ARM_LABEL = {"seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}

TASKS = ["MoleculeACE", "HIV", "BACE", "Ames", "Tox21", "QM7"]   # the paper's canonical six
LOWER_BETTER = {"MoleculeACE", "QM7"}                            # rmse; the rest are roc_auc/nef1
# BACE / Tox21 / QM7 come from the MoleculeNet CV tree, so the ablation wave has its OWN copy of
# them and the cross-wave floor check below applies. MoleculeACE / CBS / Ames are scored in shared
# benchmark trees (chemeleon_suite/, cbs_benchmark/) where the ablation families and the floor are
# siblings in ONE tree -- there is no second wave to drift against, so they are not checked.
MOLNET_TASKS = ["BACE", "Tox21", "QM7", "HIV"]

FLOOR_RUNS = ["e2e_random_00", "e2e_random_01", "e2e_random_02"]  # phase2, end2end floor
FROZEN_ABL = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN_PH2 = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["e2e_no_pretrain"]["label"]                    # "no pretrain, end2end"


def _suite_mean(run_dir, task):
    # delegated so the QM7 (and ESOL/Lipophilicity) z-scored-vs-native split is resolved in ONE
    # place for every figure -- see figures/sixpanel.NATIVE_SUBDIRS.
    return suite_run_mean(run_dir, task)


def _suite_mean_raw(run_dir, task):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    d = json.load(open(p))
    v = d.get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def _wave_mean(wave, runs, task):
    # one consistent subdir across the whole run set -- never a per-run mix
    return suite_wave_mean(wave, runs, task)


def _lift(arm_val, task, floor_val):
    return canonical_lift(arm_val, floor_val, task)


def _joint(task, arm):
    """The one MolNet subdir BOTH trees can supply for this task — see joint_molnet_subdirs.
    Without it the QM7 lift reads +99% because the floor has a native re-eval and the arm does not.
    """
    return joint_molnet_subdirs(task, [(ABL.name, [arm]), (PHASE2.name, FLOOR_RUNS)])


def _arm_value(arm, task):
    """The ablation family's value on one canonical panel.

    The six seq_* families were evaluated on MoleculeACE / CBS / Ames under the SAME directory name
    they use in the MolNet ablation tree, so one name serves all four trees.
    """
    return canonical_value(task, dict(mace=arm, mol=arm), molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, arm))


def _floor_value(task, arm):
    """The end-to-end random-init floor, read in the SAME convention as the arm it is lifted against."""
    return canonical_value(task, ARMS["e2e_no_pretrain"]["src"], molnet_subdirs=_joint(task, arm))


def compute():
    """All C2 numbers, once. Shared by the standalone figure and the assembled fig_C."""
    # cross-wave safety: the two waves' frozen random-encoder floors must agree before we borrow
    # phase2's end2end floor for the ablation wave's arms (same check as the old notebook).
    tasks, dropped = crosswave_safe(TASKS, ABL, PHASE2, FROZEN_ABL, MOLNET_TASKS)
    report_crosswave(dropped, FLOOR_LABEL)
    if not dropped:
        print(f"   frozen floors agree across both waves on all {len(MOLNET_TASKS)} MolNet tasks "
              f"-> safe to lift against phase2's {FLOOR_LABEL} floor "
              f"(MoleculeACE/CBS/Ames share ONE tree, so there is nothing to drift)")

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
    # points are coloured by eval task ONLY (user 2026-08-17): a second marker dimension for the
    # SFT family made the plot harder to interpret and duplicated what panels d-f carry anyway.
    for t in [t for t in TASKS if t in set(TK)]:
        sel = [k for k in range(len(pts)) if TK[k] == t]
        if sel:
            ax.scatter(X[sel], Y[sel], s=34, color=TASK_COLORS[t], marker="o",
                       edgecolor="white", lw=0.5, zorder=3)
    xs = np.linspace(X.min(), X.max(), 50)
    ax.plot(xs, a + b * xs, color=SHADES["random"][0], lw=STYLE["lw_thin"], ls=(0, (4, 2)),
            zorder=2)
    ax.axhline(0, color=SHADES["e2e"][0], lw=STYLE["lw_thin"], zorder=1)
    ax.set_xlabel("mean max Tanimoto: task \u2194 SFT family" if compact else
                  "mean max ECFP4 Tanimoto: eval task \u2194 SFT family   (right = more similar)")
    ax.set_ylabel(LIFT_YLABEL)
    ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)

    task_handles = [Line2D([], [], color=TASK_COLORS[t], marker="o", ls="none", ms=5,
                           mec="white", mew=0.5, label=t) for t in TASKS if t in set(TK)]
    if compact:
        # inside, lower right (empty quadrant of the scatter) so the assembled figure's two rows
        # can share one left-to-right width -- no legend hanging outside the panel.
        # BOXED, and no title (user 2026-08-19): sitting unframed on the same axes as the scatter,
        # the legend's own markers read as data points. A light frame separates key from data.
        # BLACK box, and this is the ONLY boxed legend in the set (user 2026-08-19): panel (c) is
        # the one whose legend markers are the same glyph as its data points, so it is the one that
        # needs separating. Elsewhere a frame is just clutter.
        # TWO COLUMNS (user 2026-08-19: "the legend for c) is blocking data points"). A 5-row
        # single column is ~0.65in tall, which at this panel's scale is ~18 y-units and reached up
        # to lift=-12 -- right through the HIV point at (0.400, -17.96), the lowest in the cloud.
        # 3 rows is ~0.40in, so with the deepened floor below the box clears that point by ~6
        # y-units. Checked against the data, not by eye: see the assertion after set_ylim.
        # 3 x 2 (user 2026-08-19). Each column halved is a row saved, and every row saved comes
        # straight off the forced floor below -- 2 columns needed the axis dropped to -32 to open
        # an empty band, 3 columns need only -24, so the point cloud keeps 25% more of the panel.
        ax.legend(handles=task_handles, loc="lower right", ncol=3, frameon=True, framealpha=1.0,
                  edgecolor=STYLE["ink"], facecolor="white", fontsize=FS["legend"],
                  handletextpad=0.25, borderaxespad=0.4, borderpad=0.35, labelspacing=0.25,
                  columnspacing=0.7, handlelength=0.9)
    else:
        # ALSO inside the axes. A legend anchored outside (the old bbox_to_anchor=(1.02, 1.0))
        # expands savefig's tight bbox past the page width, so this figure came out 7.10in wide
        # against the set's 6.69in and LaTeX then downscaled its fonts relative to every other
        # figure. Keep every legend inside the canvas.
        # BOXED, and no title (user 2026-08-19): sitting unframed on the same axes as the scatter,
        # the legend's own markers read as data points. A light frame separates key from data.
        # BLACK box, and this is the ONLY boxed legend in the set (user 2026-08-19): panel (c) is
        # the one whose legend markers are the same glyph as its data points, so it is the one that
        # needs separating. Elsewhere a frame is just clutter.
        ax.legend(handles=task_handles, loc="lower right", frameon=True, framealpha=1.0,
                  edgecolor=STYLE["ink"], facecolor="white", fontsize=FS["legend"],
                  handletextpad=0.3, borderaxespad=0.4, borderpad=0.45, labelspacing=0.3)

    # The lower-right legend was landing on the point cloud. Dropping the floor opens an empty
    # band beneath the data for it to sit in, rather than shrinking the legend (user 2026-08-17).
    # Only the FLOOR is forced; the top stays data-driven. -24 (user 2026-08-19): the 3x2 legend
    # is 2 rows, ~7 y-units at this scale, and the deepest point is -17.96, so the band -24..-19
    # clears it. Every row the legend loses comes straight off this number, which is why the
    # arrangement and the floor are set together rather than independently.
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -24), hi)

    ax.set_title("Transfer vs chemical similarity" if compact else
                 "Supervised pretraining: transfer vs chemical similarity",
                 loc="left" if compact else "center",
                 fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tag:
        ax.text(*((-0.09, 1.05) if compact else (-0.14, 1.02)), tag, transform=ax.transAxes,
                fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")


def main():
    data = compute()
    pts, X, Y, TK = data["pts"], data["X"], data["Y"], data["TK"]
    r, rho, p = data["r"], data["rho"], data["p"]
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 3.4))
    draw(ax, data)
    # Two lines. As one line this title measured 7.04in — WIDER than the 6.69in canvas — so
    # savefig's tight bbox grew the whole figure around it and C2 printed with smaller fonts than
    # every other figure once LaTeX scaled it back to \textwidth.
    title(ax, f"Fig C2 \u2014 Supervised pretraining: transfer vs chemical similarity\n"
              f"(n={len(pts)}, Pearson r={r:+.2f}, Spearman \u03c1={rho:+.2f}, p={p:.3f})")
    # right=0.72 dates from when the task legend hung OUTSIDE the axes; it now sits inside, so
    # that 28% reservation was pure waste. The 0.70in left overflow that used to inflate this
    # figure was the one-line TITLE, not the y-label (the title is centred on the axes and was
    # wider than the canvas) -- fixed by wrapping it above, so the left margin only needs to hold
    # the y-label.
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.085, right=0.985)
    save(fig, "fig_C2", subdir="panels")
    plt.close(fig)

    from scipy import stats as _st
    print(f"\nC2 H10: n={len(pts)} arm x task cells, Pearson r={r:+.3f}, "
          f"Spearman rho={rho:+.3f} (p={p:.3f}), similarity range {X.min():.2f}-{X.max():.2f}")
    for t in TASKS:
        m = [k == t for k in TK]
        if sum(m) >= 3:
            print(f"   within {t:<14}: rho={_st.spearmanr(X[m], Y[m]).statistic:+.2f}"
                  f" (n={sum(m)})")


if __name__ == "__main__":
    main()

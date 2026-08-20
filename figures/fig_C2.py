"""Fig C2 -- does molecular similarity (SFT data <-> eval task) explain transfer of SUPERVISED
pretraining?  The H10 test, on the deduped ablation wave.

ONE script, ONE figure: figures_v2/figC2.png / .pdf

What it shows
-------------
Every point is one (SFT family, eval task) pair (24 = 4 families x 6 tasks -- the same cells as
the old ipynb H10 panel), coloured by eval task only: x = mean max ECFP4 Tanimoto similarity between
the family's pretraining molecules and the task's molecules (figure_data/_tanimoto/
family_task_similarity.csv; families are sampled, so similarities are lower bounds), y = the
arm's lift over "no pretrain, random" -- the random-init encoder FROZEN, probed the same way the
arms are.

FLOOR CHANGED 2026-08-20 (user), from the fine-tuned random init to the frozen one. The arms here
are frozen probes, so the old floor mixed "did pretraining help" with "frozen vs fine-tuned" in
one number. Matching the floor's protocol to the arm's is the whole point; it is the same change
made in fig_C1 and fig_D, so C, D and E now share one floor and one meaning of "lift".

Arms are the molecule-set SFT families of the deduped ablation wave (figure_data/
climb_v2_ablation_dedup): PCBA, L1000 (MCF7+VCAP pooled), PCQM, and sparse_all (PCBA+L1000).
All are unsup->sup warm-started from ONE shared 2M-FP MLM base -- the wave has no random-init
single-family arms, so "supervised pretraining" here always sits on top of that shared MLM base.
The dense descriptor arms (seq_mtr, seq_dense_plus_sparse) have NO family molecule set to measure
similarity against, so they have no x-coordinate and are absent by construction (they belong to
the task-similarity figure D, not here).

The deduped wave drops 34,301 eval molecules from SFT via a blocklist, so no arm trains on
eval-test molecules. All arms and the floor are 5-fold CV means (suite_summary.json <task>_MEAN).

NOTHING IS BORROWED CROSS-WAVE ANY MORE. The old end2end floor lived in climb_v2_phase2 and had to
be imported into the ablation wave under a runtime agreement check, because the ablation wave has
no end2end runs. It DOES have its own random_baseline_0{0,1,2}, so the frozen floor is read
in-wave and arm and floor are siblings. The agreement check is kept as a drift tripwire with no
number depending on it.

Eval tasks are the canonical six (n=24 = 4 families x 6 tasks). seq_sparse_all's x pools its
three families. An earlier version of this note said Lipophilicity drops out for want of a floor;
Lipophilicity is not in the canonical panel set at all, so that sentence described a panel set
this figure no longer uses.

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

FLOOR_RUNS = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]  # frozen floor
FROZEN_ABL = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN_PH2 = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["random_encoder"]["label"]                     # "no pretrain, random"


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
    return joint_molnet_subdirs(task, [(ABL.name, [arm]), (ABL.name, FLOOR_RUNS)])


def _arm_value(arm, task):
    """The ablation family's value on one canonical panel.

    The six seq_* families were evaluated on MoleculeACE / CBS / Ames under the SAME directory name
    they use in the MolNet ablation tree, so one name serves all four trees.
    """
    return canonical_value(task, dict(mace=arm, mol=arm), molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, arm))


def _floor_value(task, arm):
    """The no-pretraining floor, read in the SAME convention as the arm it is lifted against.

    FLOOR CHANGED 2026-08-19 (user): frozen random encoder ("no pretrain, random"), not the
    fine-tuned one. The arms on this figure are FROZEN probes, so the fine-tuned floor mixed the
    pretraining question with the frozen-vs-fine-tuned question in a single lift.

    It also removes a cross-wave borrow rather than merely relabelling one. The ablation wave has
    NO end2end floor, which is why phase2's was imported under a runtime agreement check; it has
    its OWN random_baseline_0{0,1,2}. Reading the floor from ABL makes arm and floor siblings in
    one wave, so the agreement check now guards a borrow that no longer happens.
    """
    return canonical_value(task, ARMS["random_encoder"]["src"], molnet_root=ABL.name,
                           molnet_subdirs=_joint(task, arm))


def compute():
    """All C2 numbers, once. Shared by the standalone figure and the assembled fig_C."""
    # Cross-wave check, KEPT although the borrow it guarded is gone. Until 2026-08-19 the floor
    # was phase2's end2end run imported into the ablation wave, and this check licensed that
    # import by requiring the two waves' frozen random encoders to agree. The floor is now the
    # ablation wave's OWN frozen random encoder, so arm and floor are siblings and there is
    # nothing to borrow. What the check still buys is a wave-drift tripwire: if the two waves'
    # identical-by-construction floors ever diverge, something has changed underneath both
    # figures, and that is worth knowing even when no number depends on it.
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
    # "ECFP4" here is NOT the anchor's fingerprint and must not be "fixed" to match it.
    # The MODEL featurizer became Morgan r=3 counts with chirality on 2026-08-19; this
    # SIMILARITY axis deliberately stays stereo-blind binary r=2 (scripts/
    # compute_tanimoto_novelty.py), because a stereo-blind match is the LOOSER definition
    # of "the model already read this molecule" and therefore over-counts memorization --
    # the conservative direction for a null result. Two questions, two answers (user
    # 2026-08-19: "the tanimoto I'm ok with. but otherwise let's definitely use the stereo").
    ax.set_xlabel("mean max Tanimoto: task \u2194 SFT family" if compact else
                  "mean max ECFP4 Tanimoto: eval task \u2194 SFT family   (right = more similar)")
    ax.set_ylabel(LIFT_YLABEL)
    ax.grid(ls=":", lw=0.6, color=STYLE["grid"])
    ax.set_axisbelow(True)

    task_handles = [Line2D([], [], color=TASK_COLORS[t], marker="o", ls="none", ms=5,
                           mec="white", mew=0.5, label=t) for t in TASKS if t in set(TK)]
    # LEGEND PLACEMENT IS CHOSEN FROM THE DATA, not fixed to a corner.
    #
    # It was pinned to one corner with a comment recording how many points that corner held on the
    # day it was written ("lower-left contains 0 of 24"). That is a measurement, not a property:
    # changing the floor from fine-tuned to frozen moved the cloud and put a point straight under
    # the box, and the assertion -- correctly -- failed. Re-pinning to whichever corner happens to
    # be empty today would just reset the same trap for the next data change.
    #
    # So: open room below, then test all four corners against the actual box footprint and take
    # the first empty one. If none is empty the assertion fires rather than the emptiest being
    # picked -- a legend on one point is still a legend on a point.
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax0, ax1 = X.min(), X.max()
    BW, BH = (0.62, 0.20) if compact else (0.52, 0.30)   # box footprint as a fraction of each axis
    def CORNER_TESTS(ay0, ay1):
        xl = ax0 + BW * (ax1 - ax0); xr = ax1 - BW * (ax1 - ax0)
        yb = ay0 + BH * (ay1 - ay0); yt = ay1 - BH * (ay1 - ay0)
        return [("lower left",  lambda x, y: x <= xl and y <= yb),
                ("lower right", lambda x, y: x >= xr and y <= yb),
                ("upper left",  lambda x, y: x <= xl and y >= yt),
                ("upper right", lambda x, y: x >= xr and y >= yt)]
    # Open room BELOW in steps until some corner is genuinely clear. Growing the axis is the
    # cheap resolution here -- the y range is a lift percentage with no natural bound, so empty
    # space at the bottom costs nothing a reader can misread, whereas a box over a point hides
    # data. The alternative (shrinking the legend) was already spent: it is 3 columns and 2 rows.
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
    # BOXED and untitled (user 2026-08-19): on the same axes as the scatter an unframed legend's
    # markers read as data points. This is the only boxed legend in the set, for that reason.
    kw = dict(handles=task_handles, loc=loc, frameon=True, framealpha=1.0,
              edgecolor=STYLE["ink"], facecolor="white")
    if compact:
        ax.legend(ncol=3, fontsize=FS["legend"] - 0.5, handletextpad=0.25, borderaxespad=0.35,
                  borderpad=0.3, labelspacing=0.22, columnspacing=0.6, handlelength=0.9, **kw)
    else:
        ax.legend(ncol=2, fontsize=FS["legend"], handletextpad=0.3, borderaxespad=0.4,
                  borderpad=0.45, labelspacing=0.3, **kw)
    print(f"   C2 legend -> {loc} at {pad:.2f} bottom padding ("
          + ", ".join(f"{n}:{len(q)}" for n, q in occupancy) + " points per corner)")

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

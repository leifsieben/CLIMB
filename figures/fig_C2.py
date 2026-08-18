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
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, SHADES, TASK_COLORS

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

TASKS = ["BACE", "BBBP", "ESOL", "HIV", "Lipophilicity", "QM7", "Tox21"]
LOWER_BETTER = {"ESOL", "Lipophilicity", "QM7"}                  # rmse; the rest are roc_auc

FLOOR_RUNS = ["e2e_random_00", "e2e_random_01", "e2e_random_02"]  # phase2, end2end floor
FROZEN_ABL = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FROZEN_PH2 = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["e2e_no_pretrain"]["label"]                    # "no pretrain, end2end"


def _suite_mean(run_dir, task):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    d = json.load(open(p))
    v = d.get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def _wave_mean(wave, runs, task):
    vs = [_suite_mean(wave / r, task) for r in runs]
    vs = [v for v in vs if np.isfinite(v)]
    return float(np.mean(vs)) if vs else np.nan


def _lift(arm_val, task, floor_val):
    if not (np.isfinite(arm_val) and np.isfinite(floor_val)) or floor_val == 0:
        return np.nan
    if task in LOWER_BETTER:
        return 100 * (floor_val - arm_val) / abs(floor_val)
    return 100 * (arm_val - floor_val) / abs(floor_val)


def compute():
    """All C2 numbers, once. Shared by the standalone figure and the assembled fig_C."""
    # cross-wave safety: the two waves' frozen random-encoder floors must agree before we borrow
    # phase2's end2end floor for the ablation wave's arms (same check as the old notebook).
    drift = []
    for t in TASKS:
        a = _wave_mean(ABL, FROZEN_ABL, t)
        b = _wave_mean(PHASE2, FROZEN_PH2, t)
        if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 5e-3:
            drift.append((t, a, b))
    if drift:
        print("   WARNING - frozen floors drifted apart across waves; the borrowed end2end floor "
              "is no longer safe:")
        for t, a, b in drift:
            print(f"      {t}: ablation={a:.4f} vs phase2={b:.4f} ({100*abs(a-b)/abs(b):.1f}%)")
    else:
        print(f"   frozen floors agree across both waves on all {len(TASKS)} tasks "
              f"-> safe to lift against phase2's {FLOOR_LABEL} floor")

    sim = pd.read_csv(SIMP)
    pts = []
    for arm, fams in ARM2FAM.items():
        for t in TASKS:
            av = _suite_mean(ABL / arm, t)
            fv = _wave_mean(PHASE2, FLOOR_RUNS, t)
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
    for t in TASKS:
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
    ax.set_ylabel("lift (%)" if compact else f"lift over {FLOOR_LABEL} (%)")
    ax.grid(ls=":", lw=0.5, color=STYLE["grid"])
    ax.set_axisbelow(True)

    task_handles = [Line2D([], [], color=TASK_COLORS[t], marker="o", ls="none", ms=5,
                           mec="white", mew=0.5, label=t) for t in TASKS if t in set(TK)]
    if compact:
        # inside, lower right (empty quadrant of the scatter) so the assembled figure's two rows
        # can share one left-to-right width -- no legend hanging outside the panel.
        ax.legend(handles=task_handles, title="eval task", loc="lower right",
                  frameon=False, fontsize=FS["legend"], title_fontsize=FS["legend"],
                  handletextpad=0.3, borderaxespad=0.2)
    else:
        # ALSO inside the axes. A legend anchored outside (the old bbox_to_anchor=(1.02, 1.0))
        # expands savefig's tight bbox past the page width, so this figure came out 7.10in wide
        # against the set's 6.69in and LaTeX then downscaled its fonts relative to every other
        # figure. Keep every legend inside the canvas.
        ax.legend(handles=task_handles, title="eval task", loc="lower right",
                  frameon=False, fontsize=FS["legend"], title_fontsize=FS["legend"],
                  handletextpad=0.3, borderaxespad=0.2)

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
    fig, ax = plt.subplots(figsize=(STYLE["col2"], 3.4)
    draw(ax, data)
    title(ax, f"Fig C2 \u2014 Supervised pretraining: transfer vs chemical similarity"
              f"   (n={len(pts)}, Pearson r={r:+.2f}, Spearman \u03c1={rho:+.2f}, p={p:.3f})")
    fig.subplots_adjust(top=0.88, bottom=0.15, right=0.72)
    save(fig, "figC2")
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

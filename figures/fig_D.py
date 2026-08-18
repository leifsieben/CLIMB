"""Fig D -- task similarity for supervised pretraining: which SFT label type transfers WHERE?

ONE script, ONE figure: figures_v2/figD.png / .pdf

What it shows
-------------
Both analyses use the deduped ablation wave (figure_data/climb_v2_ablation_dedup): every arm is
unsup->sup warm-started from ONE shared 2M-FP MLM base, the SFT blocklist drops 34,301 eval
molecules (no arm trains on eval-test molecules), and every number is a 5-fold CV mean lifted
against the honest floor ("no pretrain, end2end" -- the random-init encoder fine-tuned on the
task; a frozen random encoder is close to automatic to beat, so it is not the comparator). The
end2end floor is borrowed cross-wave from climb_v2_phase2 only while the two waves' frozen
random-encoder floors agree -- checked at runtime, printed, never assumed.

  (a) Which SFT label type helps? Mean lift per arm (horizontal bars; right = helps, left =
      hurts), with the 2M MLM base itself and the ECFP (XGBoost) anchor for context.
  (b) Transfer matrix: SFT label family (rows) x eval task (columns), cell = lift% over the
      floor. Sparse families (PCBA, L1000 bioassay screens; PCQM quantum) vs the dense descriptor
      family (MTR: predicting ~200 RDKit descriptors from SMILES) and their union.
  (c) The task-similarity mapping, rethought (user request 2026-08-17): not only "sparse helps
      sparse" but also "does DENSE descriptor pretraining map onto descriptor-LIKE tasks?".
      Only the two canonical representatives are drawn -- dense (MTR) and sparse all; the full
      6-family version was unreadable (user decision 2026-08-17). The other families stay in
      (a) and (b). Eval tasks are grouped a priori: property regression = ESOL, QM7 (labels are
      descriptor-predictable physchem/quantum quantities -- the regime where the ECFP+desc anchor
      is strongest) vs bioassay classification = BACE, BBBP, HIV, Tox21 (context-dependent
      screens). One line per family connects its mean lift in the two groups; small markers
      are the per-task values so the group means hide nothing. A clear mapping = the dense line
      falls left-to-right (helps descriptor-like tasks relatively more), the sparse line rises.

All three panels share the family colours, so bars, matrix rows and slope lines line up.

Lipophilicity is absent everywhere: the phase2 end2end floor was never scored on it, so there is
no honest floor to lift against (same restriction as Fig C2; n_tasks: 2 property + 4 bioassay).

Run:  python3 -m figures.fig_D
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, title, check_font
from figures.arms import ARMS, SHADES

check_font()

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "figure_data" / "climb_v2_ablation_dedup"
PHASE2 = ROOT / "figure_data" / "climb_v2_phase2"

# SFT families (rows of the matrix / bars), display order
FAMILIES = ["seq_mtr", "seq_dense_plus_sparse", "seq_pcba", "seq_l1000", "seq_pcqm",
            "seq_sparse_all"]
FAM_LABEL = {"seq_mtr": "dense", "seq_dense_plus_sparse": "dense+sparse",
             "seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}
FAM_SHORT = {"seq_mtr": "dense (MTR)", "seq_dense_plus_sparse": "dense+sparse",
             "seq_pcba": "PCBA", "seq_l1000": "L1000", "seq_pcqm": "PCQM",
             "seq_sparse_all": "sparse all"}
# all arms here are unsup->sup from the shared 2M MLM base -> the u2s shade ladder, dark = dense
# families, light = sparse families
FAM_COL = {"seq_mtr": SHADES["u2s"][0], "seq_dense_plus_sparse": SHADES["u2s"][1],
           "seq_sparse_all": SHADES["u2s"][2], "seq_pcba": SHADES["u2s"][3],
           "seq_l1000": SHADES["u2s"][3], "seq_pcqm": SHADES["u2s"][4]}
FAM_MARKER = {"seq_mtr": "o", "seq_dense_plus_sparse": "s", "seq_pcba": "D",
              "seq_l1000": "v", "seq_pcqm": "^", "seq_sparse_all": "P"}
SLOPE_FAMS = ["seq_mtr", "seq_sparse_all"]          # panel (c): only these two (user decision)

TASKS = ["ESOL", "QM7", "BACE", "BBBP", "HIV", "Tox21"]          # property | bioassay
TASK_GROUP = {"ESOL": "property", "QM7": "property",
              "BACE": "bioassay", "BBBP": "bioassay", "HIV": "bioassay", "Tox21": "bioassay"}
GROUPS = ["property", "bioassay"]
GROUP_LABEL = {"property": "property\nregression", "bioassay": "bioassay\nclassification"}
GROUP_MEMBERS = "descriptor-like: ESOL, QM7          bioassay: BACE, BBBP, HIV, Tox21"
LOWER_BETTER = {"ESOL", "QM7"}                                   # rmse; the rest are roc_auc

FLOOR_RUNS = ["e2e_random_00", "e2e_random_01", "e2e_random_02"]
FROZEN = ["random_baseline_00", "random_baseline_01", "random_baseline_02"]
FLOOR_LABEL = ARMS["e2e_no_pretrain"]["label"]

# panel (a) context rows: the shared MLM base itself (phase2 wave) and the XGBoost anchor
BASE_RUN = PHASE2 / "unsup_2M"
ANCHOR_RUN = ABL / "ecfp4_anchor"
BASE_LABEL = "unsupervised"      # the (2M MLM base) detail lives in the caption, not the label
ANCHOR_LABEL = ARMS["ecfp"]["label"] + " (XGBoost)"


def _suite_mean(run_dir, task):
    p = run_dir / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return np.nan
    v = json.load(open(p)).get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


def _wave_mean(wave, runs, task):
    vs = [v for v in (_suite_mean(wave / r, task) for r in runs) if np.isfinite(v)]
    return float(np.mean(vs)) if vs else np.nan


def _lift(arm_val, task, floor_val):
    if not (np.isfinite(arm_val) and np.isfinite(floor_val)) or floor_val == 0:
        return np.nan
    if task in LOWER_BETTER:
        return 100 * (floor_val - arm_val) / abs(floor_val)
    return 100 * (arm_val - floor_val) / abs(floor_val)


def compute():
    """All D numbers, once. Shared by the standalone figure and the assembled fig_C."""
    # cross-wave floor safety (same check as Fig C2)
    drift = [(t, _wave_mean(ABL, FROZEN, t), _wave_mean(PHASE2, FROZEN, t)) for t in TASKS]
    drift = [(t, a, b) for t, a, b in drift
             if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 5e-3]
    if drift:
        print("   WARNING - frozen floors drifted apart across waves; borrowed end2end floor "
              "unsafe:")
        for t, a, b in drift:
            print(f"      {t}: ablation={a:.4f} vs phase2={b:.4f}")

    # lift matrix: family x task
    H = pd.DataFrame(index=FAMILIES, columns=TASKS, dtype=float)
    for fam in FAMILIES:
        for t in TASKS:
            H.loc[fam, t] = _lift(_suite_mean(ABL / fam, t), t, _wave_mean(PHASE2, FLOOR_RUNS, t))

    # context rows for panel (a): MLM base + XGBoost anchor, lifted against the same floor
    base_row = {t: _lift(_suite_mean(BASE_RUN, t), t, _wave_mean(PHASE2, FLOOR_RUNS, t))
                for t in TASKS}
    anchor_row = {t: _lift(_suite_mean(ANCHOR_RUN, t), t, _wave_mean(PHASE2, FLOOR_RUNS, t))
                  for t in TASKS}
    bar_rows = ([(BASE_LABEL, float(np.nanmean(list(base_row.values()))), ARMS["unsup"]["color"])]
                + [(FAM_LABEL[f], float(np.nanmean([H.loc[f, t] for t in TASKS])), FAM_COL[f])
                   for f in FAMILIES]
                + [(ANCHOR_LABEL, float(np.nanmean(list(anchor_row.values()))),
                    ARMS["ecfp"]["color"])])
    return dict(H=H, bar_rows=bar_rows)


def draw(axB, axM, axS, data, tags=("a", "b", "c"), compact=False):
    """Draw the three D panels onto existing axes (standalone or assembled context).
    compact=True raises the panel tags above the (overflowing) titles and moves the slope
    panel's group members into the tick labels so nothing spills past the column edge."""
    H, bar_rows = data["H"], data["bar_rows"]

    # --- which SFT label type helps? mean lift per arm ----------------------------------------
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
    axB.set_xlabel(f"mean lift over {FLOOR_LABEL} (%)")
    axB.grid(axis="x", ls=":", lw=0.6, color=STYLE["grid"])
    axB.set_axisbelow(True)
    axB.set_title("Lift by SFT label type",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[0]:
        axB.text(*((-0.18, 1.05) if compact else (-0.80, 1.03)), tags[0], transform=axB.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

    # --- (b) transfer matrix ------------------------------------------------------------------
    vmax = float(np.nanmax(np.abs(H.values)))
    vmax = max(20.0, np.ceil(vmax / 10) * 10)
    norm = mpl.colors.SymLogNorm(linthresh=5, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    im = axM.imshow(H.values, cmap="PuOr_r", norm=norm, aspect="auto")
    axM.set_xticks(range(len(TASKS)))
    axM.set_xticklabels([t for t in TASKS] if compact else
                        [f"{t}\n[{TASK_GROUP[t][:4]}.]" for t in TASKS],
                        fontsize=FS["annot"], rotation=30, ha="right")
    axM.set_yticks(range(len(FAMILIES)))
    axM.set_yticklabels([FAM_SHORT[f] for f in FAMILIES], fontsize=FS["annot"])
    axM.grid(False)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            if np.isfinite(v):
                axM.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=FS["annot"],
                         color="white" if abs(v) > 14 else "#222222")
    cbt = [t for t in (-vmax, -20, -5, 0, 5, 20, vmax) if abs(t) <= vmax]
    cb = axM.figure.colorbar(im, ax=axM, fraction=0.046, pad=0.03, ticks=cbt)
    cb.ax.yaxis.set_major_formatter(ticker.FixedFormatter(
        [f"{t:+.0f}".replace("+0", "0") for t in cbt]))
    cb.set_label("lift (%)" if compact else "lift (%), symlog", fontsize=FS["legend"])
    cb.ax.tick_params(labelsize=FS["annot"])
    axM.set_title("SFT family \u2192 eval task", loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[1]:
        axM.text(*((-0.14, 1.05) if compact else (-0.30, 1.03)), tags[1], transform=axM.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

    # --- (c) task-similarity mapping: dense vs sparse_all only (user 2026-08-17: the full -----
    # family set was unreadable). The two canonical representatives carry the claim; legend +
    # distinct markers (user request) instead of end-of-line text labels.
    for fam in SLOPE_FAMS:
        per_task = {g: [H.loc[fam, t] for t in TASKS if TASK_GROUP[t] == g] for g in GROUPS}
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
    axS.legend(handles=handles, loc="upper right", frameon=False, fontsize=FS["legend"],
               handletextpad=0.4)
    axS.axhline(0, color=SHADES["random"][0], lw=0.6, zorder=1)
    axS.set_xlim(-0.38, 1.38)
    axS.set_xticks([0, 1])
    if compact:
        axS.set_xticklabels(["property\n(ESOL, QM7)", "bioassay\n(BACE, BBBP,\nHIV, Tox21)"],
                            fontsize=FS["annot"])
    else:
        axS.set_xticklabels([GROUP_LABEL[g] for g in GROUPS], fontsize=FS["annot"])
        axS.set_xlabel(GROUP_MEMBERS, fontsize=FS["annot"])
    # assembled row: d's xlabel and e's colorbar already define the quantity -- no y-label here
    axS.set_ylabel("" if compact else f"mean lift over {FLOOR_LABEL} (%)")
    axS.grid(axis="y", ls=":", lw=0.6, color=STYLE["grid"])
    axS.set_axisbelow(True)
    axS.set_title("Descriptor-like task mapping",
                  loc="left" if compact else "center",
                  fontsize=FS["title"], fontweight="bold", pad=9 if compact else 4)
    if tags and tags[2]:
        axS.text(*((-0.12, 1.05) if compact else (-0.055, 1.03)), tags[2], transform=axS.transAxes,
                 fontsize=FS["panel_tag"], fontweight="bold", va="bottom", ha="left")

def main():
    data = compute()
    H, bar_rows = data["H"], data["bar_rows"]
    fig = plt.figure(figsize=(STYLE["col2"], 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.30], height_ratios=[1.0, 0.92],
                          hspace=0.55, wspace=0.60)
    draw(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]), data)
    # y=1.02 put the suptitle ABOVE the canvas, which inflates savefig's tight bbox; keep it in.
    title(fig, "Fig D \u2014 Task similarity between supervised pretraining and downstream task",
          y=0.985)
    # left/right widened to bring the panel-a y-label (was 0.38in past the canvas) and the
    # transfer-matrix colorbar (0.17in past) back inside, so savefig's tight bbox stops growing
    # this figure beyond the page width.
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.218, right=0.945)
    save(fig, "fig_D", subdir="panels")
    plt.close(fig)

    print("\nD transfer matrix (lift % over", FLOOR_LABEL + "):")
    print(H.round(1).to_string())
    print("\nD panel (a) mean lift per arm:")
    for name, v, _ in bar_rows:
        print(f"   {name:<28} {v:+6.1f}%")
    print("\nD mapping (mean lift per task group):")
    for fam in FAMILIES:
        ms = {g: float(np.nanmean([H.loc[fam, t] for t in TASKS if TASK_GROUP[t] == g]))
              for g in GROUPS}
        print(f"   {FAM_LABEL[fam]:<24} property {ms['property']:+6.1f}%   "
              f"bioassay {ms['bioassay']:+6.1f}%   gap {ms['property']-ms['bioassay']:+6.1f}")


if __name__ == "__main__":
    main()

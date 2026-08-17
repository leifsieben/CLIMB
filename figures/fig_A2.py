"""Fig A2 — per-benchmark scores for the eight headline models.

ONE script, ONE figure: figures_v2/figA2.png / .pdf

What it shows
-------------
Eight models on each of the six benchmark panels: both XGBoost anchors, the three headline CLIMB
recipes, both no-pretrain controls, and CheMeleon. **CLIMB models carry a black dotted fill**, the
XGBoost anchors and CheMeleon are solid, so the model families separate at a glance and survive
greyscale printing. Panel title line 1 is the panel name + a direction arrow (up = higher better,
down = lower better); line 2 is the number of test molecules the panel actually scores (every
molecule appears in the test fold exactly once across the 5-fold/provided-fold CV, so this is the
size of the evaluation, not the whole dataset). Sourced from the raw prediction/manifest files, not
typed by hand -- see TEST_N below.

One reference line per panel: the black dotted line is the **random-encoder level**, keyed once on
the right of the figure. Anything not clearing it is not beating an untrained network on that
benchmark. No model legend — the bars are labelled, so a legend would only restate them.

NO PANEL IS ZOOMED. Every bar starts at a meaningful floor — 0.5 (chance) on the ROC-AUC panels,
0 on RMSE and NEF1 — so bar length stays proportional to the quantity. The honest cost is that
MoleculeACE and QM7 look more compressed than the differences on them warrant.

Error bars are drawn with a WHITE HALO (matplotlib path_effects) around the black whisker/cap.
Without it, a black error bar sitting exactly at a black-edged, black-dot-hatched bar's top is
visually indistinguishable from the border -- confirmed by pixel-level zoom -- and the near-black
"random encoder" bar makes it worse still (black-on-black). The halo makes every bar's error visible
regardless of what is drawn under it.

Error bars: ±1 SD, on BACE/Tox21/QM7/CBS computed by `scripts/six_panel_aggregate.py::seed_stats`
from the fold-ENSEMBLE rows (never the `_cell` rows -- see that function's docstring for why: the
ensembled-prediction fold score is the intended point estimate, `_cell` values are a different,
strictly worse estimator that must never be averaged into the value or the error bar). Spread is
across the 3 pretraining-seed dirs where more than one exists; ECFP, ECFP+desc and CheMeleon have
exactly one dir (no pretraining stage to replicate -- a fact about those models, not a data gap),
so their bar falls back to spread across the panel's 5 folds, honestly a different but equally
real quantity. hERG uses the 3 eval seeds (Polaris was run once per model, on ONE provided split of
132 molecules, so that panel's bars understate the true uncertainty more than the others).
MoleculeACE uses the 95% target-cluster bootstrap half-width.

KNOWN BAD CELL — CheMeleon on QM7
---------------------------------
CheMeleon's QM7 RMSE is not a measurement of its representation, it is a fitting failure.
Predicting the training mean scores 228.7 on QM7 (targets are μ≈−1531, σ≈228.7), so CheMeleon is
*worse than chance*, dragged by one diverged fold while every other model is flat across the same
5 folds. Flag it in the caption or re-run that arm before quoting the number — an AWS re-run has
been requested (2026-08-16).

Run:  python3 -m figures.fig_A2
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font
from figures.arms import ARMS, PANELS, PANEL_ORDER, system, label
from figures.sixpanel import load_mainline, load_bootstrap

check_font()
INK = "#000000"
CLIMB_HATCH = "...."          # small black dots mark the CLIMB models (density x4,
                              # with hatch.linewidth 0.35 in style.py for fine dots)

# the eight models, in a fixed reading order: anchors -> CLIMB recipes -> controls -> external
MODELS = ["ecfp", "ecfp_desc", "sup_dense", "unsup", "u2s_dense",
          "random_encoder", "e2e_no_pretrain", "chemeleon_e2e"]
# 2026-08-17: was "chemeleon", a single arm labelled "end2end" but sourced from chemeleon_FROZEN --
# which is what put the frozen arm's broken QM7 value (268.8, fold2=434, worse than a constant
# predictor) on this end2end comparison. "chemeleon_e2e" is the native D-MPNN-from-foundation run,
# 3 seeds, QM7 = 198.7. See figures/arms.py for the frozen arm's genuine QM7 failure.
REFERENCE = "random_encoder"

# where a bar starts: chance level for ROC-AUC, zero otherwise. Never a zoom.
FLOOR = {"roc_auc": 0.5, "nef1": 0.0, "macro_rmse": 0.0, "rmse": 0.0}


def _err(extra):
    """±1 SD across replicate runs, parsed out of the `extra` field of mainline_8M.csv.

    A genuine 0.0 (e.g. a deterministic featurizer whose 5 fold scores happen to tie) is kept --
    it is a real result, not a signal that data is missing. Only an ABSENT key returns NaN."""
    if not isinstance(extra, str):
        return np.nan
    d = dict(kv.split("=") for kv in extra.split(";") if "=" in kv)
    for k in ("sd_seeds", "sd_evalseeds"):
        if k in d:
            try:
                return float(d[k])
            except ValueError:
                return np.nan
    return np.nan


def table():
    df = load_mainline()
    df = df[df.arm.isin(MODELS)]
    val = df.pivot_table(index="arm", columns="panel", values="value", observed=True)
    err = (df.assign(e=df.extra.map(_err))
             .pivot_table(index="arm", columns="panel", values="e", observed=True))
    boot = load_bootstrap().set_index("arm")
    for a in MODELS:                                   # MoleculeACE: half-width of the 95% CI
        if a in boot.index and np.isfinite(boot.loc[a, "ci_lo"]):
            err.loc[a, "MoleculeACE"] = (boot.loc[a, "ci_hi"] - boot.loc[a, "ci_lo"]) / 2
    return (val.reindex(index=MODELS, columns=PANEL_ORDER),
            err.reindex(index=MODELS, columns=PANEL_ORDER))


VAL, ERR = table()

# Test-set size per panel -- every molecule appears in its test fold exactly once (5-fold CV, or
# the benchmark-provided split for hERG), so this is the number of predictions the panel's metric
# is computed over. Read from the raw files, not hand-typed:
#   BACE/Tox21/QM7  unique mol_index in climb_v2_phase2/fp_desc_anchor/moleculenet_cv/test_predictions.csv
#   CBS             unique mol_index in cbs_benchmark/fp_desc_anchor/moleculenet_cv/test_predictions.csv
#   hERG            n_test in chemeleon_suite/data/polaris/polaris_manifest.json["tdcommons/herg"]
#   MoleculeACE     sum of test-split rows over the 30 chemeleon_suite/data/moleculeace/*.csv files
TEST_N = {"MoleculeACE": 9802, "CBS": 10445, "BACE": 1513, "hERG": 132, "Tox21": 7823, "QM7": 6838}


def short(a):
    """Row label. CheMeleon's recipe is called 'end2end', which would be indistinguishable from
    the 'no pretrain, end2end' control, so it is named by its system instead."""
    return "CheMeleon" if system(a) == "CheMeleon" else label(a)


def _limits(p, pad=0.06):
    v = VAL[p].dropna()
    hi = float(v.max())
    e = ERR[p].dropna()
    if len(e):
        hi = max(hi, float((VAL[p] + ERR[p]).max()))
    lo = FLOOR.get(PANELS[p]["metric"], 0.0)
    return lo, hi + (hi - lo) * pad


def build():
    fig, axes = plt.subplots(2, 3, figsize=(STYLE["col2"], 5.1))
    x = np.arange(len(MODELS))
    names = [short(a) for a in MODELS]

    for ax, p in zip(axes.ravel(), PANEL_ORDER):
        d = PANELS[p]
        lo, hi = _limits(p)

        bars = ax.bar(x, [VAL.loc[a, p] - lo for a in MODELS], bottom=lo, width=0.74,
                      color=[ARMS[a]["color"] for a in MODELS],
                      edgecolor=INK, linewidth=0.8, zorder=2)
        for b, a in zip(bars, MODELS):
            if system(a) == "CLIMB":
                b.set_hatch(CLIMB_HATCH)      # black dots; the bar keeps its black border

        # Error bars drawn as a SEPARATE call (not bar(..., yerr=...)) so each whisker/cap can
        # carry a white halo -- otherwise a black error bar disappears into a black bar border
        # or a black hatch (see the module docstring: confirmed by pixel zoom, not a hunch).
        _, caps, bcaps = ax.errorbar(x, VAL[p], yerr=ERR[p], fmt="none", ecolor=INK,
                                     elinewidth=1.0, capsize=2.2, capthick=1.1, zorder=6)
        halo = [pe.withStroke(linewidth=2.6, foreground="white")]
        for cap in caps:
            cap.set_path_effects(halo)
        for bc in bcaps:
            bc.set_path_effects(halo)

        ax.axhline(VAL.loc[REFERENCE, p], color=INK, ls=":", lw=1.1, zorder=2)

        ax.set_ylim(lo, hi)
        ax.set_ylabel(d["metric_label"], fontsize=FS["annot"], color=INK)
        arrow = "↑" if d["higher_better"] else "↓"
        n = TEST_N[p]
        ax.set_title(f"{d['label']} {arrow}\nn = {n:,} test molecules",
                     fontsize=FS["title"], fontweight="bold", color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=FS["caption"], color=INK)
        ax.grid(axis="y", ls=":", lw=0.5, color=STYLE["grid"]); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    # single key for the one reference line, parked top-right of the panels
    fig.legend(handles=[Line2D([], [], color=INK, ls=":", lw=1.2, label="random encoder")],
               loc="upper right", bbox_to_anchor=(1.0, 0.98), frameon=False,
               fontsize=FS["legend"], handlelength=1.8, labelcolor=INK)
    return fig


def main():
    print(f"{'model':34s} " + " ".join(f"{p:>12s}" for p in PANEL_ORDER))
    for a in MODELS:
        cells = " ".join(f"{VAL.loc[a,p]:12.3f}" if np.isfinite(VAL.loc[a, p]) else f"{'—':>12s}"
                         for p in PANEL_ORDER)
        print(f"{system(a) + ' · ' + label(a):34s} {cells}")
    print()
    fig = build()
    save(fig, "figA2")
    plt.close(fig)


if __name__ == "__main__":
    main()

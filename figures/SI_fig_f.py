"""SI Fig f — is the frozen-probe result a property of the EMBEDDING or of the HEAD?

ONE script, ONE figure: figures_v2/SI_fig_f.png / .pdf

LETTERED f, NOT g. The old SI fig f (the class-B resolution block) was dropped on
2026-08-19 and its content folded into fig_G, which left the SI sequence reading
a, b, c, d, e, g -- a gap a reader will hunt for. This figure takes the free letter.

Every frozen-probe number in this paper is an embedding scored through ONE head, and the headline
comparison puts an XGBoost-on-fingerprints anchor beside MLP-on-transformer-embedding arms. That
confounds two things: the representation and the classifier fitted on top of it. If the anchor wins
because gradient boosting suits 2048 sparse bits and an MLP suits a 512-d dense vector, that is a
statement about heads, not about pretraining.

So each representation is scored through BOTH heads on the same splits, seeds and folds, and drawn
as a SLOPE: left is XGBoost, right is MLP, one line per representation. The question the figure
answers is whether the lines CROSS. Parallel lines mean the head is a level shift and the ranking
of representations is head-independent -- which is what the rest of the paper assumes. Crossing
lines mean the ranking depends on the head and every frozen comparison needs re-reading.

WHICH HALF OF EACH PAIR IS NEW. Three of the four representations are normally scored with an MLP
probe and the classical anchor with XGBoost, so this figure needs the OTHER half of each pair:
    ECFP4+desc          has XGBoost (mainline)      needs MLP
    CLIMB unsupervised  has MLP (mainline)          needs XGBoost
    CLIMB supervised    has MLP (mainline)          needs XGBoost
    CheMeleon frozen    has MLP (mainline)          needs XGBoost
The existing half is read from the mainline table so it is the SAME number the other figures draw;
the new half is read from <tag>__<head>/ written by scripts/head_comparison_run.sh.

A TRAP THIS FIGURE EXISTS TO AVOID, recorded because it already cost a retraction. eval_v2 did not
standardize or median-impute the classical descriptor block for non-tree heads, so the MLP on
`fp_desc` diverged -- ESOL RMSE 4,382,073 and BBBP NaN -- and "the MLP is much worse than XGBoost"
was reported and then withdrawn. Tree heads are scale-invariant and never showed it. Fixed in
commit 0ab0388; any MLP-on-fp_desc cell produced before that is invalid, not merely noisy.

Run:  python3 -m figures.SI_fig_f
"""
from __future__ import annotations
from pathlib import Path
import csv
import statistics as st

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures.style import STYLE, FS, save, check_font, mark_empty
from figures.arms import ARMS, PANELS, PANEL_ORDER

check_font()
ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "figure_data"
INK = "#000000"

HEADS = ["xgb", "mlp"]
XTICKS = ["XGBoost", "MLP"]

# (arm key in arms.py, run tag used by head_comparison_run.sh, which head the MAINLINE half is)
#
# CheMeleon MUST BE THE FROZEN VARIANT HERE (user 2026-08-19), never chemeleon_e2e. The question
# is whether the PROBE HEAD changes the ranking of frozen representations, so every line has to be
# a frozen embedding scored through a head. chemeleon_e2e fine-tunes its whole network on each
# task, which is not a probe at all -- putting it on this axis would compare a fine-tune against
# three probes and answer nothing. It is also why audit check 7 lets this figure name CheMeleon:
# frozen-vs-frozen carries none of the protocol confound the rule exists to prevent.
SERIES = [("ecfp_desc",        "fp_desc_anchor",   "xgb"),
          ("unsup",            "unsup_8M",         "mlp"),
          ("sup_dense",        "skip_dense_8M",    "mlp"),
          ("chemeleon_frozen", "chemeleon_frozen", "mlp")]
assert all(a != "chemeleon_e2e" for a, _, _ in SERIES), \
    "SI fig f compares PROBE HEADS on frozen embeddings; chemeleon_e2e is a fine-tune, not a probe"

# metric per canonical panel, and which way is better
METRIC = {"MoleculeACE": "rmse", "HIV": "nef1", "BACE": "roc_auc",
          "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}


def _mainline():
    """{(arm, panel): value} from the table every other figure draws, so the existing half of each
    pair is literally the same number rather than a re-derivation of it."""
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


def _head_cell(tag, head, panel):
    """The new half: <tag>__<head>/moleculenet_cv, or the suite dirs for MoleculeACE/Ames.

    Returns None when the run has not been done -- the panel then says so rather than dropping the
    line silently, which is the difference between "no effect" and "not measured".
    """
    task = panel
    if panel in ("MoleculeACE", "Ames"):
        track = "moleculeace" if panel == "MoleculeACE" else "polaris"
        d = FD / "chemeleon_suite" / track / f"{tag}__{head}"
        f = d / ("results.csv" if panel == "MoleculeACE" else "polaris_scores.csv")
        if not f.exists():
            return None
        vals = []
        for r in csv.DictReader(f.open()):
            if panel == "MoleculeACE":
                if r.get("subset") == "overall" and r.get("metric") == "rmse":
                    vals.append(float(r["value"]))
            elif r.get("metric") == "roc_auc" and r.get("value") not in ("", "nan"):
                vals.append(float(r["value"]))
        return st.mean(vals) if vals else None

    f = FD / "climb_v2_phase2" / f"{tag}__{head}" / "moleculenet_cv" / "moleculenet_summary.csv"
    if not f.exists():
        return None
    m = METRIC[panel]
    vals = [float(r["main_value"]) for r in csv.DictReader(f.open())
            if r["dataset"] == task and r["main_metric"] == m
            and r["head_seed"] not in ("MEAN", "STD") and r["main_value"] not in ("", "nan")]
    return st.mean(vals) if vals else None


def compute():
    """{panel: {arm: [xgb_value, mlp_value]}}, with None for anything not run."""
    main = _mainline()
    out = {}
    for p in PANEL_ORDER:
        cells = {}
        for arm, tag, main_head in SERIES:
            pair = {}
            pair[main_head] = main.get((arm, p))
            other = "mlp" if main_head == "xgb" else "xgb"
            pair[other] = _head_cell(tag, other, p)
            cells[arm] = [pair["xgb"], pair["mlp"]]
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
            # The run has not been done. Say so ON the panel -- an empty slope panel is otherwise
            # indistinguishable from "both heads scored identically", which is the finding.
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
                      label=ARMS[a]["label"]) for a, _, _ in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052), ncol=4,
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.0, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=0.35)
    save(fig, "SI_fig_f")
    plt.close(fig)

    print("\nSI Fig f — same embedding, two probe heads (XGBoost -> MLP):\n")
    print(f"   {'panel':<13}{'representation':<22}{'XGBoost':>10}{'MLP':>10}   delta")
    for p in PANEL_ORDER:
        for arm, _, _ in SERIES:
            x, m = R[p][arm]
            if x is None or m is None:
                miss = "XGBoost" if x is None else "MLP"
                print(f"   {p:<13}{ARMS[arm]['label'][:20]:<22}{'—':>10}{'—':>10}   {miss} not run")
                continue
            print(f"   {p:<13}{ARMS[arm]['label'][:20]:<22}{x:>10.4f}{m:>10.4f}   {m - x:>+8.4f}")
    if n_missing:
        print(f"\n   {n_missing} of {len(PANEL_ORDER)} panels have NO head-comparison run yet "
              f"(scripts/head_comparison_run.sh).")


if __name__ == "__main__":
    main()

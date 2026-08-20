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
as a SLOPE: left is the MLP, right is XGBoost (user 2026-08-20), one line per representation. The question is whether the
lines CROSS. Parallel means the head is a level shift and the ranking of representations is
head-independent -- which is what the rest of the paper assumes.

THE LINES CROSS ON EVERY PANEL (6 of 6, complete 2026-08-20). The ranking of the four representations is NOT the same under the two heads
anywhere:

  MoleculeACE  ECFP4+desc leads under both heads, but CheMeleon goes from SECOND under XGBoost
               (0.688) to LAST under the MLP (0.826), crossing both CLIMB arms
  Ames         CheMeleon leads under XGBoost (0.873) and is THIRD under the MLP (0.831); both
               CLIMB arms improve under the MLP while both classical/external arms get worse
  BACE         XGBoost puts CheMeleon first, the MLP puts ECFP4+desc first
  Tox21        XGBoost puts ECFP4+desc first, the MLP puts CheMeleon first
  QM7          XGBoost puts ECFP4+desc first by 8 kcal/mol; under the MLP it is third and
               CheMeleon falls from second to LAST (195.3 -> 211.5, +16.2)
  HIV          the top two hold but the CLIMB arms swap, and CheMeleon loses 0.063 NEF1

The size of the effect is the point: CheMeleon moves 16.2 kcal/mol on QM7, 0.063 NEF1 on HIV and
0.138 macro-RMSE on MoleculeACE between heads, which is larger than most differences this paper
reports BETWEEN representations. It is the same arm every time, and always in the direction of
preferring the tree ensemble.

STATE IT ONCE AS A PROPERTY OF THE REPRESENTATION, NOT SIX TIMES AS A SURPRISE. Counting the
head swap MLP -> XGBoost panel by panel, the direction is not mixed, it is opposite by arm:

  CheMeleon frozen   XGBoost is better on 5 of 6 panels
                     MoleculeACE -0.138 macro RMSE, QM7 -16.2 kcal/mol, HIV +0.063 NEF1,
                     Ames +0.042, BACE +0.006; only Tox21 goes the other way (-0.010)
  CLIMB unsup.       XGBoost is WORSE on 5 of 6 (MoleculeACE +0.052 RMSE, QM7 +4.4, Ames -0.035,
                     HIV -0.016, Tox21 -0.012; BACE flat at +0.0003)
  CLIMB sup., desc   XGBoost is worse on 4 of 6

And it is not confined to this figure: in fig_A1 the same swap moves CheMeleon frozen from 14th of
25 to 3rd, while both CLIMB arms LOSE 2-4 positions. Five independent measurements, one direction.

The honest reading is that CheMeleon's 512-d representation suits a tree ensemble and CLIMB's does
not, which is a statement about the geometry of the two embeddings rather than about either
model's quality. It also means no single-head ranking of these representations is protocol-free:
whichever head is chosen, one family is being read through the probe that suits it least.

COMPARE DOWN A COLUMN, NEVER ACROSS THE DIAGONAL. CheMeleon-under-XGBoost (0.688 on MoleculeACE)
beats ECFP4+desc-under-the-MLP (0.738), which invites "CheMeleon is the best representation on
MoleculeACE". WITHIN the XGBoost column ECFP4+desc leads at 0.676; the cross-head pairing flatters
CheMeleon only because of the head swap this figure exists to expose. The same trap runs the other
way on Ames, where CheMeleon-XGBoost (0.873) against ECFP4+desc-MLP (0.807) looks like a wide win
and the within-column gap is 0.873 vs 0.870 -- a tie.

AND THE SHARPER SENTENCE, ON MoleculeACE, ALL UNDER XGBoost: bare ECFP4 with no descriptors reads
0.6877 against CheMeleon-frozen's 0.6875. A 2048-bit fingerprint matches a molecular foundation
model to three decimals under the same probe, and the descriptor and R3FP variants beat both
(ECFP4+desc 0.6757, R3FP 0.6721, R3FP+desc 0.6676). Ames is the counterweight and is why neither
"CheMeleon is strong" nor "classical wins" survives a single panel: there CheMeleon and ECFP4+desc
tie at the top under XGBoost while both CLIMB arms sit 0.07-0.11 below.

WHAT THAT DOES AND DOES NOT LICENCE. It does not overturn fig_A1: that figure scores every arm
through the head each is normally used with, which is the honest engineering comparison, and the
classical anchors lead it under both heads here. What it does mean is that a frozen-probe number
is a property of the PAIR (representation, head) and must not be quoted as a property of the
representation alone -- so any single-head statement of the form "embedding X beats embedding Y"
needs this figure beside it. The two arms whose ordering is most head-sensitive are exactly the two
external/classical ones, not the CLIMB pair.

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

# MLP LEFT, XGBoost RIGHT (user 2026-08-20). The pair order is defined ONCE here and everything
# downstream -- the slope's x positions, the tick labels, the printed table -- reads it, so the
# two can never disagree about which end is which.
HEADS = ["mlp", "xgb"]
XTICKS = ["MLP", "XGBoost"]

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


def _stem(arm, panel):
    """The run-dir stem for `arm` IN THE TREE THAT `panel` lives in, from arms.py.

    An arm's directory is not one name. ecfp_desc is `fp_desc_anchor` in climb_v2_phase2 and
    `fp_desc` in the suite trees, which is exactly why arms.py keeps a separate stem per tree.
    This file used to carry ONE hardcoded tag per series, so it looked for `fp_desc_anchor__mlp`
    on MoleculeACE, where the directory is `fp_desc__mlp`. The run existed and the panel printed
    "MLP not run" -- a missing-data message generated by a naming mismatch, which is worse than a
    blank because it sends someone to buy GPU time for a result already on disk.
    """
    src = ARMS[arm]["src"]
    key = "mace" if panel in ("MoleculeACE", "Ames") else "mol"
    v = src.get(key) or src.get("mol")
    return v[0] if isinstance(v, (list, tuple)) else v


def _head_cell(arm, head, panel):
    """The new half: <stem>__<head>/moleculenet_cv, or the suite dirs for MoleculeACE/Ames.

    Returns None when the run has not been done -- the panel then says so rather than dropping the
    line silently, which is the difference between "no effect" and "not measured".
    """
    # THE POLARIS FILE IS THE WHOLE TRACK, NOT ONE PANEL. polaris_scores.csv holds 28 tasks; the
    # Ames panel is ONE of them, PANELS["Ames"]["polaris_task"] = "tdcommons/ames". Until
    # 2026-08-20 this branch filtered on metric alone and averaged every roc_auc row in the file,
    # i.e. nine unrelated Polaris classification tasks pooled into a cell labelled Ames. It read
    # 0.7688 where Ames alone is 0.7652 -- close enough to look like a plausible Ames number,
    # which is why it survived. arms.py already declared the task; this file simply never asked.
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

    # The table follows HEADS, so it always reads left-to-right in the same order as the slope.
    lo, hi = (XTICKS[0], XTICKS[1])
    print(f"\nSI Fig f — same embedding, two probe heads ({lo} -> {hi}):\n")
    print(f"   {'panel':<13}{'representation':<22}{lo:>10}{hi:>10}   delta")
    for p in PANEL_ORDER:
        for arm, _, _ in SERIES:
            a0, a1 = R[p][arm]
            if a0 is None or a1 is None:
                miss = lo if a0 is None else hi
                print(f"   {p:<13}{ARMS[arm]['label'][:20]:<22}{'—':>10}{'—':>10}   {miss} not run")
                continue
            print(f"   {p:<13}{ARMS[arm]['label'][:20]:<22}{a0:>10.4f}{a1:>10.4f}   {a1 - a0:>+8.4f}")
    if n_missing:
        print(f"\n   {n_missing} of {len(PANEL_ORDER)} panels have NO head-comparison run yet "
              f"(scripts/head_comparison_run.sh).")


if __name__ == "__main__":
    main()

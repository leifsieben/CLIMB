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

THE HEAD PREFERENCE IS REPRESENTATION-DEPENDENT, and that is the whole finding. Delta is
XGBoost minus MLP, signed so that better-under-XGBoost is positive:

                        MolACE     HIV    BACE    Ames   Tox21     QM7    prefers
  ECFP4+desc            +.0645  +.0192  -.0027  +.0620  +.0077  +10.02    XGBoost, 5 of 6
  CLIMB unsupervised    -.0298  -.0425  -.0043  -.0046  -.0150   -4.15    MLP,     6 of 6
  CLIMB supervised      -.0198  -.0113  +.0079  -.0017  -.0170   -3.61    MLP,     5 of 6

A 2048-bit sparse fingerprint plus 217 descriptors suits a tree ensemble; a 512-d dense embedding
suits an MLP. Neither is a small effect where it matters: ECFP4+desc gains 0.065 macro RMSE on
MoleculeACE and 10.0 kcal/mol on QM7 by switching head, which is wider than the gap between any
two representations on those panels under a single head.

THIS IS THE EVIDENCE FOR fig_A's PROBE RULE. fig_A scores each representation at the head that
suits it -- ECFP4 arms at XGBoost, every CLM at frozen+MLP -- rather than forcing one head on
all. That rule is only defensible because the preference is measured and representation-dependent
rather than assumed, and this figure is the measurement. Forcing a single head does not remove the
confound, it just picks which representation to handicap.

WHAT THAT DOES AND DOES NOT LICENCE. A frozen-probe number is a property of the PAIR
(representation, head) and must not be quoted as a property of the representation alone, so any
single-head statement of the form "embedding X beats embedding Y" needs this figure beside it. It
does not overturn fig_A: the classical anchor leads under BOTH heads on five of the six panels
here, so its position is not an artefact of the head it is given.

THE EXPOSURE THIS FIGURE CANNOT CLOSE. It measures three representations. fig_A also ranks three
literature CLMs (ChemBERTa-2, MoLFormer, SELFIES-TED) at frozen+MLP that have NEVER been measured
at XGBoost, so their head is assumed to suit them rather than measured to.
notes/figA-seed-axis-is-not-uniform.md records that decision and its magnitude.

WHICH HALF OF EACH PAIR IS NEW. Two of the three representations are normally scored with an MLP
probe and the classical anchor with XGBoost, so this figure needs the OTHER half of each pair:
    ECFP4+desc          has XGBoost (mainline)      needs MLP
    CLIMB unsupervised  has MLP (mainline)          needs XGBoost
    CLIMB supervised    has MLP (mainline)          needs XGBoost
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

from figures.style import STYLE, FS, save, check_font, mark_empty, row_ncol, LEGEND_BOX
from figures.arms import ARMS, series_label, PANELS, PANEL_ORDER, RETIRED

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
# EVERY LINE MUST BE A FROZEN EMBEDDING SCORED THROUGH A HEAD. The question is whether the probe
# head changes the ranking of frozen representations, so a fine-tuned arm on this axis would
# compare a fine-tune against probes and answer nothing -- which is what the assert below tests.
# The RETIRED filter is what removed CheMeleon (2026-08-23); the entry stays in the literal so the
# filter has something to act on and so the line is not silently re-added by someone reading a
# shorter list. `chemeleon_suite` further down is the shared RESULTS TREE, not the arm.
SERIES = [s for s in [("ecfp_desc",        "fp_desc_anchor",   "xgb"),
                      ("unsup",            "unsup_8M",         "mlp"),
                      ("sup_dense",        "skip_dense_8M",    "mlp"),
                      ("chemeleon_frozen", "chemeleon_frozen", "mlp")]
          if s[0] not in RETIRED]
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

    # EACH LINE IS A REPRESENTATION, so the legend names the representation and NOT its probe --
    # the probe is the x-axis. series_label() is what strips it: arms.py's labels are written for
    # fig_A1's two-line rows, where system() supplies the model name above and the label may pin a
    # probe below. Used raw here, such a label would both omit the model and contradict the axis.
    handles = [Line2D([], [], color=ARMS[a]["color"], marker="o", ms=5.0, lw=1.4,
                      label=series_label(a)) for a, _, _ in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.052), ncol=row_ncol(handles),
               fontsize=FS["legend"], handletextpad=0.5, labelspacing=0.3, columnspacing=1.4,
               borderpad=0.30, **LEGEND_BOX, labelcolor=INK)
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
                print(f"   {p:<13}{series_label(arm)[:20]:<22}{'—':>10}{'—':>10}   {miss} not run")
                continue
            print(f"   {p:<13}{series_label(arm)[:20]:<22}{a0:>10.4f}{a1:>10.4f}   {a1 - a0:>+8.4f}")
    if n_missing:
        print(f"\n   {n_missing} of {len(PANEL_ORDER)} panels have NO head-comparison run yet "
              f"(scripts/head_comparison_run.sh).")


if __name__ == "__main__":
    main()

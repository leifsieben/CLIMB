"""Loaders for the canonical 6-panel results + the ranking maths shared by Fig A1 / Table A2.

All numbers come from figure_data/six_panel/, produced by scripts/six_panel_aggregate.py.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from figures.arms import ARMS, ARM_ORDER, PANELS, PANEL_ORDER

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "figure_data" / "six_panel"


def load_mainline() -> pd.DataFrame:
    """arm x panel point estimates at the 8M budget."""
    df = pd.read_csv(SP / "mainline_8M.csv")
    df["arm"] = pd.Categorical(df["arm"], ARM_ORDER, ordered=True)
    df["panel"] = pd.Categorical(df["panel"], PANEL_ORDER, ordered=True)
    return df


def load_long() -> pd.DataFrame:
    """Replicate-level values: per target (MoleculeACE) / per seed x fold (MoleculeNet)."""
    return pd.read_csv(SP / "mainline_8M_long.csv")


def load_bootstrap() -> pd.DataFrame:
    return pd.read_csv(SP / "mainline_8M_bootstrap.csv")


def score_matrix(arms=None) -> pd.DataFrame:
    """Wide matrix of raw scores: rows = arms, columns = the 6 panels (NaN where not run)."""
    arms = arms or ARM_ORDER
    df = load_mainline()
    m = df.pivot_table(index="arm", columns="panel", values="value", observed=True)
    return m.reindex(index=arms, columns=PANEL_ORDER)


def rank_table(arms=None) -> pd.DataFrame:
    """Per-panel rank (1 = best) rescaled to a common 1..N scale, plus the mean rank.

    Rescaling matters because a panel where only k of the N arms were run would otherwise hand
    those k arms artificially good ranks (1..k instead of 1..N). Rank within the panel, then map
    [1..k] -> [1..N] linearly.

    Returns rows = arms, columns = the 6 panels + mean_rank, se_rank, n_panels, worst, best.
    """
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    N = len(arms)
    R = pd.DataFrame(index=S.index, columns=PANEL_ORDER, dtype=float)
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        r = col.rank(ascending=PANELS[p]["higher_better"] is False)   # 1 = best
        k = len(col)
        R.loc[r.index, p] = 1 + (N - 1) * (r - 1) / (k - 1) if k > 1 else 1.0
    R["n_panels"] = R[PANEL_ORDER].notna().sum(axis=1)
    R["mean_rank"] = R[PANEL_ORDER].mean(axis=1)
    R["se_rank"] = R[PANEL_ORDER].std(axis=1, ddof=1) / np.sqrt(R["n_panels"])
    R["best"] = R[PANEL_ORDER].min(axis=1)
    R["worst"] = R[PANEL_ORDER].max(axis=1)
    return R.sort_values("mean_rank")


def shortfall_table(arms=None) -> pd.DataFrame:
    """Per-panel % behind that panel's best model, plus the mean and its SE across panels.

    The effect-size counterpart to rank_table(): ranking treats a panel where the field spans
    1.8% of the metric (BBBP) exactly like one where it spans 22% (MoleculeACE), which flatters
    models that lose narrowly on compressed panels. This keeps the size of the gap.
    """
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    G = {}
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        best = col.max() if PANELS[p]["higher_better"] else col.min()
        G[p] = 100 * (best - col).abs() / abs(best)
    G = pd.DataFrame(G).reindex(index=arms, columns=PANEL_ORDER)
    G["n_panels"] = G[PANEL_ORDER].notna().sum(axis=1)
    G["mean_gap"] = G[PANEL_ORDER].mean(axis=1)
    G["se_gap"] = G[PANEL_ORDER].std(axis=1, ddof=1) / np.sqrt(G["n_panels"])
    G["best"] = G[PANEL_ORDER].min(axis=1)
    G["worst"] = G[PANEL_ORDER].max(axis=1)
    return G.sort_values("mean_gap")


def wins(arms=None) -> pd.DataFrame:
    """Per-arm win counts: how often it is the best / top-3 arm on a panel (raw, un-rescaled)."""
    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    out = {}
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        out[p] = col.rank(ascending=PANELS[p]["higher_better"] is False)
    R = pd.DataFrame(out)
    return pd.DataFrame({
        "n_panels": R.notna().sum(axis=1),
        "n_best": (R == 1).sum(axis=1),
        "n_top3": (R <= 3).sum(axis=1),
    })


def fmt_value(panel: str, v: float) -> str:
    """Format a raw score the way its panel wants it."""
    if not np.isfinite(v):
        return "—"
    return f"{v:.1f}" if panel == "QM7" else f"{v:.3f}"


def arm_labels(index) -> list:
    return [ARMS[a]["label"] if a in ARMS else a for a in index]


def arm_colors(index) -> list:
    return [ARMS[a]["color"] if a in ARMS else "#999999" for a in index]


# --------------------------------------------------------------------------- unit resolution ---
# Some regression tasks are stored in TWO conventions in the phase-2 wave: the z-scored metric
# (QM7 ~0.85) for most runs and native units (QM7 ~200 kcal/mol) for a few. The re-evaluations
# write native values to their own subdirs, one run at a time, so at any moment a wave is part
# migrated. Everything that reads a suite_summary must therefore (a) PREFER the native subdir and
# (b) never pool two subdirs into one mean -- resolving per-run instead of per-arm is what produced
# a QM7 mean of 129.9 for `no pretrain, end2end` (10 native folds averaged with 5 z-scored ones).
# See figures/README.md, "Units: never pool two subdirs of one arm".
# task -> subdirs in PREFERENCE order. The mechanism is "prefer the re-evaluated copy, and never
# pool two of them"; the reason for the re-eval differs by task. QM7/ESOL/Lipophilicity are UNIT
# re-evals (native vs z-scored). Tox21 is a VINTAGE re-eval: the 2026-08-05 missing-label fix was
# applied to the predictions but the per-fold summary rows were only partly rewritten (fold0 only,
# an interrupted re-run), so moleculenet_cv_tox21fixed/ holds rows re-scored from each run's own
# predictions. See scripts/rescore_tox21.py.
NATIVE_SUBDIRS = {
    "QM7": ("moleculenet_cv_qm7native", "moleculenet_cv"),
    "ESOL": ("moleculenet_cv_regnative", "moleculenet_cv"),
    "Lipophilicity": ("moleculenet_cv_regnative", "moleculenet_cv"),
    "Tox21": ("moleculenet_cv_tox21fixed", "moleculenet_cv"),
}


def suite_subdir(wave, runs, task):
    """The ONE subdir to read `runs` from for `task`: first candidate that any run has."""
    cands = NATIVE_SUBDIRS.get(task, ("moleculenet_cv",))
    for sub in cands:
        if any((wave / r / sub / "suite_summary.json").exists() for r in runs):
            return sub, [r for r in runs if (wave / r / sub / "suite_summary.json").exists()]
    return cands[-1], list(runs)


_SUBDIR_WARNED = set()


def suite_wave_mean(wave, runs, task, verbose=True, subdir=None):
    """Mean of <task>_MEAN over `runs`, all read from a single consistent subdir.

    `subdir` forces the choice — used when this wave is being compared against another that may
    have been re-evaluated to a different extent (see joint_molnet_subdirs).
    """
    import json
    if subdir is None:
        sub, usable = suite_subdir(wave, list(runs), task)
    else:
        sub = subdir
        usable = [r for r in runs if (wave / r / sub / "suite_summary.json").exists()]
    key = (str(wave), task, tuple(runs))
    if verbose and len(usable) < len(runs) and key not in _SUBDIR_WARNED:
        _SUBDIR_WARNED.add(key)
        skipped = [r for r in runs if r not in usable]
        print(f"   SUBDIR SKIP {task}: reading {sub}/ from {len(usable)} run(s); dropped "
              f"{','.join(skipped)} (no {sub}/ yet -- pooling them would MIX UNITS)")
    vals = []
    for r in usable:
        p = wave / r / sub / "suite_summary.json"
        if not p.exists():
            continue
        v = json.load(open(p)).get(f"{task}_MEAN")
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def suite_run_mean(run_dir, task):
    """<task>_MEAN for ONE run dir, preferring its native subdir."""
    sub, _ = suite_subdir(run_dir.parent, [run_dir.name], task)
    import json
    p = run_dir / sub / "suite_summary.json"
    if not p.exists():
        return np.nan
    v = json.load(open(p)).get(f"{task}_MEAN")
    return float(v) if v is not None else np.nan


# ---------------------------------------------------------------------------------------------
# CANONICAL-SIX VALUE RESOLUTION (added 2026-08-18 for fig_C2 / fig_D)
#
# fig_C2 and fig_D lift an arm over the end-to-end floor on each of the paper's six panels. Three
# of those panels do NOT live in the MoleculeNet CV tree at all:
#
#   MoleculeACE  figure_data/chemeleon_suite/moleculeace/<dir>/results.csv
#   Ames         figure_data/chemeleon_suite/polaris/<dir>/polaris_scores.csv
#   CBS          figure_data/cbs_benchmark/<dir>/moleculenet_cv/
#   BACE/Tox21/QM7  figure_data/<molnet_root>/<dir>/moleculenet_cv[_qm7native]/
#
# Rather than write a second implementation of those four readers, this DELEGATES to
# scripts/six_panel_aggregate.py — the module that builds every other figure's table. Reimplementing
# would be the single most likely way to reintroduce a bug we have already fixed there (the
# `_cell`-vs-fold-ensemble point estimate, the qm7native unit split, the CBS-from-a-stale-summary
# path). If the aggregator's definition of a panel value ever changes, these figures change with it.
# ---------------------------------------------------------------------------------------------
import importlib.util as _ilu
import statistics as _st

_spec = _ilu.spec_from_file_location("_six_panel_aggregate", ROOT / "scripts" / "six_panel_aggregate.py")
_spa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_spa)

CANON_METRIC = {"MoleculeACE": "macro_rmse", "CBS": "nef1", "BACE": "roc_auc",
                "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse"}
CANON_LOWER_BETTER = {"MoleculeACE", "QM7"}


def _molnet_mean(dirs, dataset, metric, root, subdirs):
    """Mean over fold-ENSEMBLE cells, falling back to per-dir suite_summary means."""
    cells = _spa.mol_fold_values(dirs, dataset, metric, root=root, subdirs=subdirs)
    if cells:
        return float(_st.mean(v for _, v in cells))
    summ = _spa.mol_dir_summaries(dirs, dataset, metric, root=root, subdirs=subdirs)
    return float(_st.mean(m for _, m, _ in summ)) if summ else float("nan")


# Tasks whose alternate subdir is a CORRECTION (a different, better answer) rather than a
# CONVENTION (the same answer in other units). The distinction decides what to do when only some
# trees have it:
#   convention (QM7 native, ESOL regnative) -> fall back to the shared convention. A lift is exactly
#       scale-invariant, so comparing two z-scored values gives the same number as two native ones.
#   correction (Tox21)                      -> DROP the task. A stale value is not a rescaled value:
#       the missing-label error moves arms by +0.015...+0.032, non-uniformly, so a lift of two
#       stale numbers is simply the wrong lift, not the right one in other units.
CORRECTION_TASKS = {"Tox21"}


def joint_molnet_subdirs(task, groups):
    """ONE MolNet subdir that EVERY (root, dirs) group can supply, for a cross-tree comparison.

    A lift is scale-invariant — 100*(k*floor - k*arm)/|k*floor| is k-free — but ONLY when the arm
    and the floor are in the same convention. canonical_value() on its own prefers each side's
    native subdir independently, which is right for an absolute panel and WRONG for a lift across
    two trees that have been re-evaluated to different extents.

    Concretely (2026-08-18): the phase-2 e2e floor has a native QM7 re-eval, the ablation wave's
    seq_* arms do not. Reading each side's preferred subdir gives 100*(205.3 - 0.90)/205.3 = +99.6%
    lift out of thin air. Resolving jointly picks the z-scored subdir BOTH trees have, and the
    lift comes out identical to what a native-native comparison would give. Nothing needs re-running.

    `groups` is [(root_name, dirs), ...]. Returns a subdirs tuple for the aggregator's readers.
    """
    from scripts.six_panel_aggregate import FD as _FD  # noqa: F401  (same FD, via the loaded module)
    cands = NATIVE_SUBDIRS.get(task, ("moleculenet_cv",))
    have = [sub for sub in cands
            if all(any((_spa.FD / root / d / sub).exists() for d in dirs) for root, dirs in groups)]
    if task in CORRECTION_TASKS:
        # ONLY the corrected subdir will do. Falling back to the shared stale one would give a lift
        # of two wrong numbers -- internally consistent and still wrong, because the correction is
        # not a uniform factor. If every tree already lacks the corrected copy the task is simply
        # uncorrected everywhere, which is a different (and currently non-existent) situation.
        return (cands[0],) if cands[0] in have else None
    return (have[0],) if have else (cands[-1],)


def canonical_value(task, src, molnet_root="climb_v2_phase2", molnet_subdirs=None):
    """Point estimate for ONE (arm, canonical panel), resolved exactly as the aggregator does.

    `src` is an arms.py-style dict: `mace` names the dir under chemeleon_suite/ (MoleculeACE and
    Ames), `mol` names the dir(s) under the MolNet tree AND under cbs_benchmark/ — the CBS runs are
    named identically to the MolNet ones, which is why one field serves both. Either may be a str
    or an explicit list. Returns nan when the panel has not been evaluated for this arm.
    """
    mace, mol = src.get("mace"), src.get("mol")
    if task == "MoleculeACE":
        pt = _spa.mace_per_target(mace) if mace else None
        if not (pt and pt.get("overall")):
            return float("nan")
        return float(_st.mean(pt["overall"].values()))
    if task == "Ames":
        cells = _spa.polaris_cells(mace, "tdcommons/ames", "roc_auc") if mace else []
        return float(_st.mean(v for _, v in cells)) if cells else float("nan")
    if not mol:
        return float("nan")
    dirs = list(mol) if isinstance(mol, (list, tuple)) else [mol]
    if task == "CBS":
        return _molnet_mean(dirs, "cbs", "nef1", "cbs_benchmark", _spa.DEFAULT_SUBDIRS)
    if molnet_subdirs is not None:
        subs = molnet_subdirs
    else:
        subs = _spa.QM7_SUBDIRS if task == "QM7" else _spa.DEFAULT_SUBDIRS
    return _molnet_mean(dirs, task, CANON_METRIC[task], molnet_root, subs)


def canonical_lift(arm_val, floor_val, task):
    """% improvement of `arm_val` over `floor_val`, sign-corrected for lower-is-better panels.

    Scale-invariant in the panel's units, so a task stored in two conventions cannot distort it —
    PROVIDED both arguments come from the same convention, which canonical_value() guarantees by
    resolving one subdir for a whole run set.
    """
    import numpy as _np
    if not (_np.isfinite(arm_val) and _np.isfinite(floor_val)) or floor_val == 0:
        return float("nan")
    d = (floor_val - arm_val) if task in CANON_LOWER_BETTER else (arm_val - floor_val)
    return 100.0 * d / abs(floor_val)


# ---------------------------------------------------------------------------------------------
# CROSS-WAVE LIFT SAFETY (added 2026-08-18)
#
# fig_C2 and fig_D lift ABLATION-wave arms over the PHASE-2 end-to-end floor, because the ablation
# wave has no end2end floor of its own. That borrow is only valid while the two waves evaluate a
# task identically. The test is the FROZEN random-encoder floor, which both waves DO have: it is
# the same three encoder inits scored by each wave's own pipeline, so any difference is protocol,
# not model.
#
# On 2026-08-18 this caught a live one. BACE is byte-identical across the two waves for all three
# inits (0.8083 / 0.8067 / 0.8173) — the same runs — but Tox21 reads 0.7710 in the ablation wave
# against 0.7526 in phase 2, a +2.4% offset, and CBS-style nef1 on the same runs differs too
# (0.5065 vs 0.4375). Same checkpoints, two different Tox21 evaluations. Lifting an ablation-wave
# Tox21 value over the phase-2 floor charges that protocol offset to pretraining: it accounts for
# 34-72% of every seq_* family's Tox21 lift. A task that fails this test is DROPPED from the
# figure, not silently drawn — an offset that size is indistinguishable from the effect being
# measured.
# ---------------------------------------------------------------------------------------------
def crosswave_safe(tasks, abl_wave, ph2_wave, frozen_runs, molnet_tasks, tol=0.01):
    """Split `tasks` into (safe, dropped) by cross-wave frozen-floor agreement.

    Only `molnet_tasks` are testable: the other canonical panels (MoleculeACE / CBS / Ames) are
    scored in shared benchmark trees where the ablation families and the floor are siblings in ONE
    tree, so there is no second wave to drift against and nothing to check.

    `tol` is a RELATIVE tolerance on the frozen floor. Returns (safe, [(task, abl, ph2, pct)]).
    """
    import numpy as _np
    safe, dropped = [], []
    for t in tasks:
        if t not in molnet_tasks:
            safe.append(t)
            continue
        subs = joint_molnet_subdirs(t, [(abl_wave.name, frozen_runs), (ph2_wave.name, frozen_runs)])
        if subs is None:
            # a corrected re-score exists for one tree and not the other: not comparable at all
            dropped.append((t, float("nan"), float("nan"), float("inf")))
            continue
        a = suite_wave_mean(abl_wave, frozen_runs, t, verbose=False, subdir=subs[0])
        b = suite_wave_mean(ph2_wave, frozen_runs, t, verbose=False, subdir=subs[0])
        if not (_np.isfinite(a) and _np.isfinite(b)) or b == 0:
            safe.append(t)
            continue
        pct = abs(a - b) / abs(b)
        (dropped.append((t, a, b, 100 * pct)) if pct > tol else safe.append(t))
    return safe, dropped


def report_crosswave(dropped, floor_label):
    """Print the drop list in a form that cannot be mistaken for a warning to be ignored."""
    if not dropped:
        return
    print(f"   DROPPED {len(dropped)} panel(s) — the two waves do NOT evaluate them the same way, "
          f"so a lift over the borrowed {floor_label} floor would charge a protocol offset to "
          f"pretraining:")
    for t, a, b, pct in dropped:
        if not np.isfinite(pct):
            print(f"      {t}: one wave has the CORRECTED re-score and the other does not — its "
                  f"predictions are pre-fix, so it cannot be re-scored from disk at all")
        else:
            print(f"      {t}: ablation frozen floor={a:.4f} vs phase2={b:.4f}  ({pct:.1f}% apart)")

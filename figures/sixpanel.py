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

    arms = arms or ARM_ORDER
    S = score_matrix(arms)
    N = len(arms)
    R = pd.DataFrame(index=S.index, columns=PANEL_ORDER, dtype=float)
    for p in PANEL_ORDER:
        col = S[p].dropna()
        if col.empty:
            continue
        r = col.rank(ascending=PANELS[p]["higher_better"] is False)
        k = len(col)
        R.loc[r.index, p] = 1 + (N - 1) * (r - 1) / (k - 1) if k > 1 else 1.0
    R["n_panels"] = R[PANEL_ORDER].notna().sum(axis=1)
    R["mean_rank"] = R[PANEL_ORDER].mean(axis=1)
    R["se_rank"] = R[PANEL_ORDER].std(axis=1, ddof=1) / np.sqrt(R["n_panels"])
    R["best"] = R[PANEL_ORDER].min(axis=1)
    R["worst"] = R[PANEL_ORDER].max(axis=1)
    return R.sort_values("mean_rank")


def shortfall_table(arms=None) -> pd.DataFrame:

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


NATIVE_SUBDIRS = {


    "QM7": ("moleculenet_cv_qm7clamped", "moleculenet_cv_qm7native", "moleculenet_cv"),
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


import importlib.util as _ilu
import statistics as _st

_spec = _ilu.spec_from_file_location("_six_panel_aggregate", ROOT / "scripts" / "six_panel_aggregate.py")
_spa = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_spa)

CANON_METRIC = {"MoleculeACE": "macro_rmse", "HIV": "nef1", "BACE": "roc_auc",
                "Ames": "roc_auc", "Tox21": "roc_auc", "QM7": "rmse", "CBS": "nef1"}
CANON_LOWER_BETTER = {"MoleculeACE", "QM7"}


def _molnet_mean(dirs, dataset, metric, root, subdirs):
    """Mean over fold-ENSEMBLE cells, falling back to per-dir suite_summary means."""
    cells = _spa.mol_fold_values(dirs, dataset, metric, root=root, subdirs=subdirs)
    if cells:
        return float(_st.mean(v for _, v in cells))
    summ = _spa.mol_dir_summaries(dirs, dataset, metric, root=root, subdirs=subdirs)
    return float(_st.mean(m for _, m, _ in summ)) if summ else float("nan")


CORRECTION_TASKS = {"Tox21"}


def joint_molnet_subdirs(task, groups):

    from scripts.six_panel_aggregate import FD as _FD
    cands = NATIVE_SUBDIRS.get(task, ("moleculenet_cv",))

    def _usable(root, d, sub):

        d_ = _spa.FD / root / d / sub
        return d_.exists() and not (d_ / "reference_scoring.json").exists()

    have = [sub for sub in cands
            if all(any(_usable(root, d, sub) for d in dirs) for root, dirs in groups)]
    if task in CORRECTION_TASKS:


        return (cands[0],) if cands[0] in have else None
    return (have[0],) if have else (cands[-1],)


def canonical_value(task, src, molnet_root="climb_v2_phase2", molnet_subdirs=None):

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

    import numpy as _np
    if not (_np.isfinite(arm_val) and _np.isfinite(floor_val)) or floor_val == 0:
        return float("nan")
    d = (floor_val - arm_val) if task in CANON_LOWER_BETTER else (arm_val - floor_val)
    return 100.0 * d / abs(floor_val)


def crosswave_safe(tasks, abl_wave, ph2_wave, frozen_runs, molnet_tasks, tol=0.01):

    import numpy as _np
    safe, dropped = [], []
    for t in tasks:
        if t not in molnet_tasks:
            safe.append(t)
            continue
        subs = joint_molnet_subdirs(t, [(abl_wave.name, frozen_runs), (ph2_wave.name, frozen_runs)])
        if subs is None:

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

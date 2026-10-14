from __future__ import annotations
import csv, math, os, sys, collections, statistics as st
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_models import _scaffold, _metric_over_cols
from figures.arms import ARMS

N_BOOT = 2000

POLARIS_PANEL, POLARIS_TASK = "Ames", "tdcommons/ames"
POLARIS_NPOS, POLARIS_NNEG = 777, 680
FD = ROOT / "figure_data"
MOL = {"BACE": ("auc", "climb_v2_phase2"), "Tox21": ("auc", "climb_v2_phase2"),
       "QM7": ("rmse", "climb_v2_phase2"), "HIV": ("nef1", "climb_v2_phase2")}


A2_ARMS = ["ecfp", "ecfp_desc", "sup_dense", "unsup", "u2s_dense",
           "random_encoder", "e2e_no_pretrain", "unsup_e2e", "sup_dense_e2e"]


SUBDIRS = {"QM7": ("moleculenet_cv_qm7native", "moleculenet_cv")}


VINTAGE_SUBDIRS = {"Tox21": "moleculenet_cv_tox21fixed"}


def _has_rows(root, run, dataset, sub):
    """True iff <root>/<run>/<sub>/ actually holds per-molecule rows for `dataset`."""
    for cand in (run, f"{run}_s0"):
        if any((FD / root / cand / sub / n).exists() for n in DUMP_NAMES):
            return load_oof(root, cand, dataset, sub) is not None
    return False


def _oof_subdir(root, runs, dataset):
    """One subdir for the whole arm: the first candidate any of its dirs has. (chosen, usable)."""
    vsub = VINTAGE_SUBDIRS.get(dataset)
    if vsub:
        import six_panel_aggregate as _spa
        keep = [r for r in runs if _spa._usable_dir(root, r, vsub)]


        runs = keep or runs
    cands = SUBDIRS.get(dataset, ("moleculenet_cv",))
    for sub in cands:


        usable = [r for r in runs if _has_rows(root, r, dataset, sub)]
        if usable:
            return sub, usable
    return cands[-1], list(runs)


DUMP_NAMES = ("test_predictions.csv", "test_predictions_gapfill.csv")


def load_oof(root, run, dataset, subdir="moleculenet_cv"):
    want = "cbs" if root == "cbs_benchmark" else dataset
    for name in DUMP_NAMES:
        p = FD / root / run / subdir / name
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        d = d[d.dataset == want]
        if not len(d):
            continue


        for c in ("y_true", "y_pred", "mol_index", "output_index"):
            if c in d.columns:
                before = d[c].notna().sum()
                d[c] = pd.to_numeric(d[c], errors="coerce")
                assert d[c].notna().sum() == before, (
                    f"{root}/{run}/{subdir}/{name}: {before - int(d[c].notna().sum())} "
                    f"non-numeric value(s) in {c} for {want} -- the dump is malformed, not just "
                    f"concatenated")
        return d
    return None


PER_DIR_SUBDIRS = {"Tox21": (("moleculenet_cv_tox21fixed", "moleculenet_cv"), 77_864)}


def load_oof_all(root, runs, dataset):

    out = []
    per_dir = PER_DIR_SUBDIRS.get(dataset) if root != "cbs_benchmark" else None
    if per_dir:
        cands, want_rows = per_dir
        for run in runs:
            for cand in (run, f"{run}_s0"):
                got = None
                for sub in cands:
                    d = load_oof(root, cand, dataset, sub)


                    if d is not None and len(d) == want_rows:
                        got = d
                        break
                if got is not None:
                    got = got.copy(); got["_dir"] = cand; out.append(got); break
        return pd.concat(out, ignore_index=True) if out else None
    subdir, runs = _oof_subdir(root, list(runs), dataset)
    for run in runs:
        for cand in (run, f"{run}_s0"):
            d = load_oof(root, cand, dataset, subdir)
            if d is not None:
                d = d.copy(); d["_dir"] = cand; out.append(d); break
    return pd.concat(out, ignore_index=True) if out else None


def fold_ids(root, smiles, y):

    if root == "cbs_benchmark":
        m = {r["smiles"]: int(r["fold"]) for r in csv.DictReader(open(ROOT / "data" / "cbs.csv"))}
        return np.array([m.get(s, -1) for s in smiles])
    import evaluate
    folds = evaluate._scaffold_kfold_indices(list(smiles), 5, 0, labels=y)
    out = np.full(len(smiles), -1)
    for j, idx in enumerate(folds):
        out[np.asarray(idx, dtype=int)] = j
    return out


def pooled_metric(m, kind, dirs, folds):
    """The BAR's estimator: mean over seed dirs of (mean over folds of the per-fold metric)."""
    per_dir = []
    for d in np.unique(dirs):
        sel = dirs == d
        vals = []
        for f in np.unique(folds[sel]):
            if f < 0:
                continue
            sub = m[sel & (folds == f)]
            if len(sub) == 0:
                continue
            v = _metric_over_cols(sub, "y_pred_a", kind)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            per_dir.append(float(np.mean(vals)))
    return float(np.mean(per_dir)) if per_dir else np.nan


def clamp_regression(m, kind, folds):
    """Clip y_pred to each fold's TRAIN target range +-25%; identity for classification metrics."""
    if kind not in ("rmse", "macro_rmse"):
        return m
    import evaluate
    out = m["y_pred_a"].to_numpy(float).copy()
    y = m["y_true_a"].to_numpy(float)
    for f in np.unique(folds):
        if f < 0:
            continue
        te = folds == f
        tr = (folds != f) & (folds >= 0)
        if not te.any() or not tr.any():
            continue
        out[te] = evaluate._bound_ood(out[te], y[tr], "regression")
    return m.assign(y_pred_a=out)


def scaffold_ci(d, kind, root, seed=0):

    m = d.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    dirs = m["_dir"].to_numpy() if "_dir" in m.columns else np.array(["one"] * len(m))


    key = "mol_index" if "mol_index" in m.columns else "raw_smiles_a"
    ref = max(np.unique(dirs), key=lambda d: m.loc[dirs == d, key].nunique())
    sub = m.loc[dirs == ref]
    uniq = sub.drop_duplicates(subset=[key]).sort_values(key)
    folds_u = fold_ids(root, uniq["raw_smiles_a"].tolist(), uniq["y_true_a"].to_numpy())
    fmap = dict(zip(uniq["raw_smiles_a"], folds_u))
    folds = np.array([fmap.get(s, -1) for s in m["raw_smiles_a"]])
    scaf = m["raw_smiles_a"].map(_scaffold).to_numpy()
    groups = collections.defaultdict(list)
    for pos, s in enumerate(scaf):
        groups[s].append(pos)
    keys = list(groups); idx = {k: np.array(v) for k, v in groups.items()}; K = len(keys)
    rng = np.random.default_rng(seed)
    m = clamp_regression(m, kind, folds)
    obs = pooled_metric(m, kind, dirs, folds)
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[keys[i]] for i in rng.integers(0, K, K)])
        v = pooled_metric(m.iloc[rows], kind, dirs[rows], folds[rows])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return obs, np.nan, np.nan, K
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi), K


def _expand(base):

    if isinstance(base, (list, tuple)):
        return list(base)
    return [base, f"{base}_s1", f"{base}_s2"]


def mace_ci(base, seed=0):
    """95% CI by resampling the 30 targets (pooled over pretraining-seed dirs + eval seeds)."""
    dirs = [d for d in _expand(base)
            if (FD / "chemeleon_suite" / "moleculeace" / d / "results.csv").exists()]
    if not dirs:
        return None
    per = collections.defaultdict(list)
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "moleculeace" / d / "results.csv")):
            if r["metric"] == "rmse" and r["subset"] == "overall":
                per[r["task"]].append(float(r["value"]))
    m = {t: st.mean(v) for t, v in per.items()}
    keys = list(m); rng = np.random.default_rng(seed)
    boots = sorted(float(np.mean([m[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
                   for _ in range(N_BOOT))


    return st.mean(m.values()), boots[int(.025*N_BOOT)], boots[int(.975*N_BOOT)], len(keys), len(dirs)


def herg_se(base):

    dirs = [d for d in _expand(base)
            if (FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv").exists()]
    vals = []
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv")):
            if r["task"] == POLARIS_TASK and r["metric"] == "roc_auc":
                vals.append(float(r["value"]))
    if not vals:
        return None
    A = st.mean(vals)


    n_dirs = len(dirs)
    n1, n0 = POLARIS_NPOS, POLARIS_NNEG
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A*(1-A) + (n1-1)*(Q1-A*A) + (n0-1)*(Q2-A*A)) / (n1*n0))
    return A, A - 1.96*se, A + 1.96*se, se, len(vals), n_dirs


def main(only=None):

    rows = []
    for arm in (only or A2_ARMS):
        spec = ARMS.get(arm)
        if not spec:
            continue
        src = spec["src"]

        for panel, (kind, root) in MOL.items():
            runs = src.get("mol") or []
            if root == "cbs_benchmark":
                runs = src.get("mol") or []
            got = load_oof_all(root, [r for r in runs if r], panel)
            if got is None:
                rows.append(dict(arm=arm, panel=panel, metric=kind, value="", ci_lo="", ci_hi="",
                                 se="", method="MISSING_OOF", n_units=0, n_dirs=0)); continue
            v, lo, hi, K = scaffold_ci(got, kind, root)


            nd = int(got["_dir"].nunique()) if "_dir" in got.columns else 1
            rows.append(dict(arm=arm, panel=panel, metric=kind, value=round(v, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4), se=round((hi-lo)/3.92, 4),
                             method="scaffold_cluster_bootstrap", n_units=K, n_dirs=nd))
            print(f"  {arm:16s} {panel:12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} scaffolds, "
                  f"{nd} seed dir(s))", flush=True)

        if src.get("mace"):
            r = mace_ci(src["mace"])
            if r:
                v, lo, hi, K, nd = r
                rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse", value=round(v,4),
                                 ci_lo=round(lo,4), ci_hi=round(hi,4), se=round((hi-lo)/3.92,4),
                                 method="target_cluster_bootstrap", n_units=K, n_dirs=nd))
                print(f"  {arm:16s} {'MoleculeACE':12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} targets)", flush=True)

        r = herg_se(src.get("mace") or "")
        if r:
            v, lo, hi, se, n, nd = r
            rows.append(dict(arm=arm, panel=POLARIS_PANEL, metric="roc_auc", value=round(v,4),
                             ci_lo=round(lo,4), ci_hi=round(hi,4), se=round(se,4),
                             method="analytic_hanley_mcneil_DERIVED", n_units=1457, n_dirs=nd))
            print(f"  {arm:16s} {POLARIS_PANEL:12s} {v:.4f} [{lo:.4f},{hi:.4f}] SE={se:.4f} (derived)", flush=True)
    out = FD / "six_panel" / "a2_errorbars.csv"
    if only and out.exists():
        keep = [r for r in csv.DictReader(out.open()) if r["arm"] not in set(only)]
        for r in keep:
            r.setdefault("n_dirs", "")

        order = {a: i for i, a in enumerate(A2_ARMS)}
        rows = sorted(keep + rows, key=lambda r: order.get(r["arm"], len(order)))
        print(f"  merged: recomputed {len(only)} arm(s), carried {len(keep)} existing rows through")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm","panel","metric","value","ci_lo","ci_hi","se","method","n_units","n_dirs"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


def _acquire_lock(force=False):

    lock = ROOT / "figure_data" / "six_panel" / ".a2_bootstrap.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except (ValueError, IndexError):
            pid = None
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if alive and not force:
            raise SystemExit(
                f"another a2_bootstrap_errorbars.py is running (pid {pid}, lock {lock}).\n"
                f"Two runs race on a2_errorbars.csv and the LAST writer wins, which can silently "
                f"publish an older data vintage under a fresh timestamp. Wait for it, or pass "
                f"--force if you are certain that pid is not this job.")
        if not alive:
            print(f"  stale lock from dead pid {pid}; taking it")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return lock


if __name__ == "__main__":
    import argparse, atexit, os, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", help="comma-separated subset of A2_ARMS to recompute and merge")
    ap.add_argument("--force", action="store_true", help="take the lock even if another run holds it")
    a = ap.parse_args()
    _lock = _acquire_lock(force=a.force)
    atexit.register(lambda: _lock.exists() and _lock.unlink())
    main(only=[x for x in a.arms.split(",") if x] if a.arms else None)

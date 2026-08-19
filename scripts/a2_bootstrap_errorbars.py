"""A2 error bars on ONE estimand: sampling uncertainty over evaluation units.

Why (user decision 2026-08-18): the previous `sd_total` was uniform in FORMULA but not in MEANING.
It measures run-to-run reproducibility on a FIXED dataset, and what it contains differs per panel --
15 cells (3 pretraining seeds x 5 folds) on the MolNet/CBS panels, but only 3 eval seeds of a
PRE-AVERAGED 30-target mean on MoleculeACE, and only head-seed noise on ONE 132-molecule split for
hERG. Identical-looking whiskers therefore encoded different questions, and the panel with the least
information (hERG) drew among the tightest bars.

This computes instead "how much would this number move under a fresh draw of the evaluation units?",
which is the quantity that governs replicability, is the same estimand in every panel, and matches
the paper's OWN A1 rigor protocol (2026-08-05 cluster-bootstrap CI).

  BACE / Tox21 / QM7 / CBS  scaffold cluster bootstrap over per-molecule OOF (Bemis-Murcko clusters)
  MoleculeACE               target cluster bootstrap over the 30 ChEMBL targets
  hERG                      CANNOT be resampled -- Polaris withholds test labels, so
                            polaris_scores.csv has no y_true. Analytic Hanley-McNeil SE instead,
                            flagged `derived` so the caption can say so.

Writes figure_data/six_panel/a2_errorbars.csv:
  arm,panel,metric,value,ci_lo,ci_hi,se,method,n_units
"""
from __future__ import annotations
import csv, math, os, sys, collections, statistics as st
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compare_models import _scaffold, _metric_over_cols            # noqa: E402
from figures.arms import ARMS                                       # noqa: E402

N_BOOT = 2000
# Polaris panel: Ames since 2026-08-18 (was hERG). n_test=1457 at the 53.32% train active rate.
POLARIS_PANEL, POLARIS_TASK = "Ames", "tdcommons/ames"
POLARIS_NPOS, POLARIS_NNEG = 777, 680
FD = ROOT / "figure_data"
MOL = {"BACE": ("auc", "climb_v2_phase2"), "Tox21": ("auc", "climb_v2_phase2"),
       "QM7": ("rmse", "climb_v2_phase2"), "HIV": ("nef1", "climb_v2_phase2")}
A2_ARMS = ["ecfp", "ecfp_desc", "sup_dense", "unsup", "u2s_dense",
           "random_encoder", "e2e_no_pretrain", "chemeleon_e2e"]


# QM7's phase-2 predictions are z-scored for most runs and native for a few; the native re-eval
# writes to moleculenet_cv_qm7native/ and exists only for the runs that needed it. Same rule as
# scripts/six_panel_aggregate: PREFER the native subdir, and never pool the two -- an arm whose
# CI mixed z-scored and native OOF would produce a meaningless interval, exactly as a mixed point
# estimate produced the bogus 129.9 QM7 mean for `no pretrain, end2end`.
SUBDIRS = {"QM7": ("moleculenet_cv_qm7native", "moleculenet_cv")}


def _oof_subdir(root, runs, dataset):
    """One subdir for the whole arm: the first candidate any of its dirs has. (chosen, usable)."""
    cands = SUBDIRS.get(dataset, ("moleculenet_cv",))
    for sub in cands:
        usable = [r for r in runs
                  if (FD / root / r / sub / "test_predictions.csv").exists()
                  or (FD / root / f"{r}_s0" / sub / "test_predictions.csv").exists()]
        if usable:
            return sub, usable
    return cands[-1], list(runs)


def load_oof(root, run, dataset, subdir="moleculenet_cv"):
    p = FD / root / run / subdir / "test_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.dataset == ("cbs" if root == "cbs_benchmark" else dataset)]
    return d if len(d) else None


def load_oof_all(root, runs, dataset):
    """OOF from EVERY pretraining-seed dir, tagged by dir. The bar pools all seeds, so the CI must
    too -- using one dir made the interval describe a different estimator than the bar."""
    out = []
    subdir, runs = _oof_subdir(root, list(runs), dataset)
    for run in runs:
        for cand in (run, f"{run}_s0"):
            d = load_oof(root, cand, dataset, subdir)
            if d is not None:
                d = d.copy(); d["_dir"] = cand; out.append(d); break
    return pd.concat(out, ignore_index=True) if out else None


def fold_ids(root, smiles, y):
    """Per-molecule fold membership, so a bootstrap draw can recompute the BAR's estimator
    (mean over folds of the per-fold metric) rather than one global ranking over pooled OOF.
    That difference is large for NEF1: CBS unsup was 0.814 globally vs 0.735 as mean-of-per-fold."""
    if root == "cbs_benchmark":
        m = {r["smiles"]: int(r["fold"]) for r in csv.DictReader(open(ROOT / "data" / "cbs.csv"))}
        return np.array([m.get(s, -1) for s in smiles])
    import eval_v2
    folds = eval_v2._scaffold_kfold_indices(list(smiles), 5, 0, labels=y)
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


def scaffold_ci(d, kind, root, seed=0):
    """95% CI by resampling whole Bemis-Murcko scaffold clusters, recomputing the SAME pooled
    estimator the bar reports (all seed dirs; per-fold then averaged)."""
    m = d.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    dirs = m["_dir"].to_numpy() if "_dir" in m.columns else np.array(["one"] * len(m))
    # Fold assignment must be computed on UNIQUE MOLECULES. Tox21's OOF carries one row per
    # (molecule, output_index) -- 12 rows per molecule -- and feeding that duplicated SMILES list to
    # _scaffold_kfold_indices produced a partition that only approximated the real one (residual
    # ~1.3% on Tox21 after the pooling fix). Deduplicate on mol_index first.
    d0 = dirs == dirs[0]
    sub = m.loc[d0]
    key = "mol_index" if "mol_index" in sub.columns else "raw_smiles_a"
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
    """<base> + its _s1/_s2 replicates -- unless arms.py already spelled the dirs out as a LIST
    (random_encoder's are _00/_01/_02, s2u_dense's are _s0/_s1/_s2)."""
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
    return st.mean(m.values()), boots[int(.025*N_BOOT)], boots[int(.975*N_BOOT)], len(keys)


def herg_se(base):
    """Hanley-McNeil analytic SE for the POLARIS panel. Polaris withholds test labels, so this is
    the one panel that cannot be resampled -- flagged DERIVED so the caption can say so.
    2026-08-18: the panel moved hERG (n=132) -> Ames (n=1457). Ames has ~4x the effective sample and
    ~2.2x the headroom per SE, which is why the swap was made."""
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
    n1, n0 = POLARIS_NPOS, POLARIS_NNEG
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A*(1-A) + (n1-1)*(Q1-A*A) + (n0-1)*(Q2-A*A)) / (n1*n0))
    return A, A - 1.96*se, A + 1.96*se, se, len(vals)


def main(only=None):
    """`only` = a subset of A2_ARMS to recompute; their rows REPLACE the matching rows in the
    existing CSV and every other arm is carried through untouched. Added 2026-08-18: the full
    sweep is ~1h, and a single arm's inputs change whenever one more replicate lands (here,
    e2e_random_02's native QM7), so recomputing all eight to refresh one is pure waste.
    """
    rows = []
    for arm in (only or A2_ARMS):
        spec = ARMS.get(arm)
        if not spec:
            continue
        src = spec["src"]
        # MolNet-shaped panels + CBS
        for panel, (kind, root) in MOL.items():
            runs = src.get("mol") or []
            if root == "cbs_benchmark":
                runs = src.get("mol") or []
            got = load_oof_all(root, [r for r in runs if r], panel)
            if got is None:
                rows.append(dict(arm=arm, panel=panel, metric=kind, value="", ci_lo="", ci_hi="",
                                 se="", method="MISSING_OOF", n_units=0)); continue
            v, lo, hi, K = scaffold_ci(got, kind, root)
            rows.append(dict(arm=arm, panel=panel, metric=kind, value=round(v, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4), se=round((hi-lo)/3.92, 4),
                             method="scaffold_cluster_bootstrap", n_units=K))
            print(f"  {arm:16s} {panel:12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} scaffolds)", flush=True)
        # MoleculeACE
        if src.get("mace"):
            r = mace_ci(src["mace"])
            if r:
                v, lo, hi, K = r
                rows.append(dict(arm=arm, panel="MoleculeACE", metric="macro_rmse", value=round(v,4),
                                 ci_lo=round(lo,4), ci_hi=round(hi,4), se=round((hi-lo)/3.92,4),
                                 method="target_cluster_bootstrap", n_units=K))
                print(f"  {arm:16s} {'MoleculeACE':12s} {v:.4f} [{lo:.4f},{hi:.4f}] ({K} targets)", flush=True)
        # hERG
        r = herg_se(src.get("mace") or "")
        if r:
            v, lo, hi, se, n = r
            rows.append(dict(arm=arm, panel=POLARIS_PANEL, metric="roc_auc", value=round(v,4),
                             ci_lo=round(lo,4), ci_hi=round(hi,4), se=round(se,4),
                             method="analytic_hanley_mcneil_DERIVED", n_units=1457))
            print(f"  {arm:16s} {POLARIS_PANEL:12s} {v:.4f} [{lo:.4f},{hi:.4f}] SE={se:.4f} (derived)", flush=True)
    out = FD / "six_panel" / "a2_errorbars.csv"
    if only and out.exists():
        keep = [r for r in csv.DictReader(out.open()) if r["arm"] not in set(only)]
        # preserve the canonical A2_ARMS order rather than appending the recomputed arms at the end
        order = {a: i for i, a in enumerate(A2_ARMS)}
        rows = sorted(keep + rows, key=lambda r: order.get(r["arm"], len(order)))
        print(f"  merged: recomputed {len(only)} arm(s), carried {len(keep)} existing rows through")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm","panel","metric","value","ci_lo","ci_hi","se","method","n_units"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", help="comma-separated subset of A2_ARMS to recompute and merge")
    a = ap.parse_args()
    main(only=[x for x in a.arms.split(",") if x] if a.arms else None)

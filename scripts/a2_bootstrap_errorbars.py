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
FD = ROOT / "figure_data"
MOL = {"BACE": ("auc", "climb_v2_phase2"), "Tox21": ("auc", "climb_v2_phase2"),
       "QM7": ("rmse", "climb_v2_phase2"), "CBS": ("nef1", "cbs_benchmark")}
A2_ARMS = ["ecfp", "ecfp_desc", "sup_dense", "unsup", "u2s_dense",
           "random_encoder", "e2e_no_pretrain", "chemeleon_e2e"]


def load_oof(root, run, dataset):
    p = FD / root / run / "moleculenet_cv" / "test_predictions.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.dataset == ("cbs" if root == "cbs_benchmark" else dataset)]
    return d if len(d) else None


def scaffold_ci(d, kind, seed=0):
    """95% CI by resampling whole Bemis-Murcko scaffold clusters (same routine as A1)."""
    m = d.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    scaf = m["raw_smiles_a"].map(_scaffold).to_numpy()
    groups = collections.defaultdict(list)
    for pos, s in enumerate(scaf):
        groups[s].append(pos)
    keys = list(groups); idx = {k: np.array(v) for k, v in groups.items()}; K = len(keys)
    rng = np.random.default_rng(seed)
    obs = _metric_over_cols(m, "y_pred_a", kind)
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[keys[i]] for i in rng.integers(0, K, K)])
        v = _metric_over_cols(m.iloc[rows], "y_pred_a", kind)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return obs, np.nan, np.nan, K
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi), K


def mace_ci(base, seed=0):
    """95% CI by resampling the 30 targets (pooled over pretraining-seed dirs + eval seeds)."""
    dirs = [d for d in (base, f"{base}_s1", f"{base}_s2")
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
    """Hanley-McNeil analytic SE. Polaris withholds test labels so this CANNOT be resampled."""
    dirs = [d for d in (base, f"{base}_s1", f"{base}_s2")
            if (FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv").exists()]
    vals = []
    for d in dirs:
        for r in csv.DictReader(open(FD / "chemeleon_suite" / "polaris" / d / "polaris_scores.csv")):
            if r["task"] == "tdcommons/herg" and r["metric"] == "roc_auc":
                vals.append(float(r["value"]))
    if not vals:
        return None
    A = st.mean(vals)
    n1, n0 = 89, 43                      # 132 test molecules at the train active ratio (67.7%)
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt((A*(1-A) + (n1-1)*(Q1-A*A) + (n0-1)*(Q2-A*A)) / (n1*n0))
    return A, A - 1.96*se, A + 1.96*se, se, len(vals)


def main():
    rows = []
    for arm in A2_ARMS:
        spec = ARMS.get(arm)
        if not spec:
            continue
        src = spec["src"]
        # MolNet-shaped panels + CBS
        for panel, (kind, root) in MOL.items():
            runs = src.get("mol") or []
            if root == "cbs_benchmark":
                runs = [src.get("cbs_dir") or (src.get("mol") or [None])[0]]
            got = None
            for run in runs:
                if not run:
                    continue
                for cand in (run, f"{run}_s0"):
                    d = load_oof(root, cand, panel)
                    if d is not None:
                        got = d; break
                if got is not None:
                    break
            if got is None:
                rows.append(dict(arm=arm, panel=panel, metric=kind, value="", ci_lo="", ci_hi="",
                                 se="", method="MISSING_OOF", n_units=0)); continue
            v, lo, hi, K = scaffold_ci(got, kind)
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
            rows.append(dict(arm=arm, panel="hERG", metric="roc_auc", value=round(v,4),
                             ci_lo=round(lo,4), ci_hi=round(hi,4), se=round(se,4),
                             method="analytic_hanley_mcneil_DERIVED", n_units=132))
            print(f"  {arm:16s} {'hERG':12s} {v:.4f} [{lo:.4f},{hi:.4f}] SE={se:.4f} (derived)", flush=True)
    out = FD / "six_panel" / "a2_errorbars.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm","panel","metric","value","ci_lo","ci_hi","se","method","n_units"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

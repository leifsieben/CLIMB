"""Cluster-bootstrap + BH-FDR 'best model' analysis for the A1 tables, both split schemes.

One rigorous test everywhere (README §8.1): a scaffold cluster-bootstrap of the paired metric
difference (resample whole Bemis-Murcko scaffolds, so repetition/clustering is corrected), with
BH-FDR across the family. Replaces the old molecule-level Wilcoxon/DeLong and the hold-out 1-sigma
rule. Writes analysis/rigor/best_model_bootstrap.csv; the notebook just reads it (the bootstrap is
too slow to run inline, esp. HIV=41k).

Two families, matching how the tables are used:
  * family="full"  : all 15 A1 arms, CV only -> drives the A1.b summary "not distinguishable from
    leader (FDR<0.05)" column and the beats-no_pretrain columns.
  * family="matrix": the 6 headline models, BOTH schemes -> the split-cell matrix. FDR is over this
    smaller pre-specified family, so the hold-out resolves whatever it legitimately can.
Hold-out (moleculenet/) also gets a single-model scaffold-bootstrap CI per arm for the A1.a
estimates+CI table (no ranking there -- the adversarial split is underpowered by design).

Decisions:
  cobest (not distinguishable from leader) = the (arm-leader) difference is NOT significant at
    BH-FDR q>=0.05 (leader trivially included). beats no_pretrain = q<0.05 AND in the arm's favour.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))          # compare_models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

import numpy as np
import pandas as pd
from compare_models import cluster_bootstrap_diff, bh_fdr, _scaffold, _metric_over_cols

DATA = Path("figure_data/climb_v2_phase2")
N_BOOT = 1000
SCHEMES = {"cv": "moleculenet_cv", "holdout": "moleculenet"}
# (task, higher_better, metric, suite_key, boot_kind)
TASKS = [("ESOL", False, "rmse", "ESOL", "rmse"), ("QM7", False, "rmse", "QM7", "rmse"),
         ("BBBP", True, "roc_auc", "BBBP", "auc"), ("BACE", True, "roc_auc", "BACE", "auc"),
         ("Tox21", True, "roc_auc", "Tox21", "auc"), ("HIV", True, "nef1", "HIV_nef1", "nef1")]

# arm -> seed runs (mirror notebook cell 10 ARMS)
SUP = ["dense", "sparse_all", "dense_plus_sparse", "minimol_full", "mixed"]
ARM_RUNS = {"ecfp4": ["ecfp4_anchor"], "fp_desc": ["fp_desc_anchor"],
            # External comparator CheMeleon (e2e), 3 seeds. runs[0] drives the bootstrap; the 3 dirs
            # give the seed spread. No hold-out run -> holdout preds are None -> A1.a stays n.d.
            "chemeleon_e2e": ["chemeleon_e2e", "chemeleon_e2e_s1", "chemeleon_e2e_s2"],
            "no_pretrain": ["random_baseline_00", "random_baseline_01", "random_baseline_02"],
            "no_pretrain_e2e": ["e2e_random_00", "e2e_random_01", "e2e_random_02"],
            "unsup_only": ["unsup_8M", "unsup_8M_s1", "unsup_8M_s2"]}
for _r in SUP:
    ARM_RUNS[f"sup_only:{_r}"] = [f"skip_{_r}_8M", f"skip_{_r}_8M_s1", f"skip_{_r}_8M_s2"]
    ARM_RUNS[f"unsup2sup:{_r}"] = [f"u2s_{_r}_from8M", f"u2s_{_r}_from8M_s1", f"u2s_{_r}_from8M_s2"]
ARM_RUNS = {a: [r for r in rs if (DATA / r).exists()] for a, rs in ARM_RUNS.items()}
ARM_RUNS = {a: rs for a, rs in ARM_RUNS.items() if rs}

FULL_ARMS = [a for a in ARM_RUNS if a not in ()]                  # all arms present
MATRIX_ARMS = ["fp_desc", "ecfp4", "unsup_only", "sup_only:dense_plus_sparse",
               "sup_only:mixed", "no_pretrain_e2e"]

_PRED = {}
def preds(run, scheme, task):
    key = (run, scheme, task)
    if key not in _PRED:
        p = DATA / run / SCHEMES[scheme] / "test_predictions.csv"
        if not p.exists(): _PRED[key] = None
        else:
            d = pd.read_csv(p)
            d = d[d.dataset == task].drop_duplicates(["dataset", "mol_index", "output_index"])
            _PRED[key] = d if len(d) else None
    return _PRED[key]

def arm_preds(arm, scheme, task):                 # seed-0 run's per-molecule predictions
    rs = ARM_RUNS[arm]
    return preds(rs[0], scheme, task) if rs else None

def arm_point(arm, scheme, task, suite_key):      # ranking value = mean over pretraining seeds
    vals = []
    for r in ARM_RUNS[arm]:
        p = DATA / r / SCHEMES[scheme] / "suite_summary.json"
        if p.exists():
            v = json.loads(p.read_text()).get(suite_key + "_MEAN")
            if v is not None and np.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def _merge(a, b):
    m = a.merge(b, on=["dataset", "mol_index", "output_index"], suffixes=("_a", "_b"))
    return m if len(m) else None

def _cbd(a, b, kind, hb, seed=0):
    m = _merge(a, b)
    if m is None: return dict(ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan)
    return cluster_bootstrap_diff(m, kind, hb, n_boot=N_BOOT, seed=seed)

def _single_ci(a, kind, seed=0):
    """Scaffold cluster-bootstrap 95% CI of ONE model's metric (for the hold-out estimates)."""
    if a is None: return (np.nan, np.nan, np.nan)
    m = a.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "raw_smiles": "raw_smiles_a"})
    scaf = m["raw_smiles_a"].map(_scaffold).to_numpy()
    groups = {}
    for pos, s in enumerate(scaf): groups.setdefault(s, []).append(pos)
    keys = list(groups); idx = {k: np.array(v) for k, v in groups.items()}; K = len(keys)
    rng = np.random.default_rng(seed)
    obs = _metric_over_cols(m, "y_pred_a", kind)
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[keys[i]] for i in rng.integers(0, K, K)])
        v = _metric_over_cols(m.iloc[rows], "y_pred_a", kind)
        if np.isfinite(v): vals.append(v)
    if not vals: return (obs, np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (obs, float(lo), float(hi))


def run_family(family, arms, schemes):
    rows = []
    for scheme in schemes:
        for task, hb, metric, sk, kind in TASKS:
            pts = {a: arm_point(a, scheme, task, sk) for a in arms}
            live = {a: v for a, v in pts.items() if np.isfinite(v) and arm_preds(a, scheme, task) is not None}
            if not live: continue
            leader = (min if not hb else max)(live, key=live.get)
            lead_p = arm_preds(leader, scheme, task)
            for a in arms:
                ap = arm_preds(a, scheme, task)
                row = dict(scheme=scheme, family=family, task=task, metric=metric, arm=a,
                           point=pts.get(a, np.nan), is_leader=(a == leader),
                           leader=leader, ci_lo=np.nan, ci_hi=np.nan,
                           vl_delta=np.nan, vl_ci_lo=np.nan, vl_ci_hi=np.nan, vl_boot_p=np.nan,
                           bf_boot_p=np.nan, bf_ci_lo=np.nan, be_boot_p=np.nan, be_ci_lo=np.nan)
                if ap is None:
                    rows.append(row); continue
                if scheme == "holdout":                       # single-model CI for the A1.a table
                    _, row["ci_lo"], row["ci_hi"] = _single_ci(ap, kind)
                # A1.a has NO ranking, so skip the (slow) full-family pairwise on the hold-out;
                # only the matrix family needs hold-out co-best. CV needs both families.
                do_pairwise = not (family == "full" and scheme == "holdout")
                if do_pairwise and a != leader:               # vs leader
                    r = _cbd(ap, lead_p, kind, hb)
                    row.update(vl_delta=r.get("boot_diff", np.nan), vl_ci_lo=r["ci_lo"],
                               vl_ci_hi=r["ci_hi"], vl_boot_p=r["boot_p"])
                if do_pairwise and family == "full":          # beats no_pretrain (CV table only)
                    for base, pfx in (("no_pretrain", "bf"), ("no_pretrain_e2e", "be")):
                        bp = arm_preds(base, scheme, task)
                        if bp is not None and a != base:
                            r = _cbd(ap, bp, kind, hb)
                            row[f"{pfx}_boot_p"] = r["boot_p"]; row[f"{pfx}_ci_lo"] = r["ci_lo"]
                rows.append(row)
            print(f"  [{family}/{scheme}] {task}: leader={leader} ({len(arms)} arms)", flush=True)
    df = pd.DataFrame(rows)
    # BH-FDR within this family (over all vs-leader tests), and within the beats tests
    for col, q in (("vl_boot_p", "vl_boot_q"), ("bf_boot_p", "bf_boot_q"), ("be_boot_p", "be_boot_q")):
        if col in df and df[col].notna().any():
            df[q] = np.nan
            mask = df[col].notna()
            df.loc[mask, q] = bh_fdr(df.loc[mask, col].to_numpy())
    return df


def main():
    out = Path("analysis/rigor"); out.mkdir(parents=True, exist_ok=True)
    full = run_family("full", FULL_ARMS, ["cv", "holdout"])
    # CLIMB-only family (drop the two XGBoost anchors): drives the "CLIMB-only x/n" column, i.e.
    # co-best AMONG CLIMB arms relative to the best CLIMB arm. CV only (that column is CV).
    climb_arms = [a for a in FULL_ARMS if a not in ("ecfp4", "fp_desc", "chemeleon_e2e")]
    climb = run_family("climb", climb_arms, ["cv"])
    matrix = run_family("matrix", MATRIX_ARMS, ["cv", "holdout"])
    df = pd.concat([full, climb, matrix], ignore_index=True)
    # decisions
    df["cobest"] = df.is_leader | (df.get("vl_boot_q", pd.Series(np.nan, index=df.index)) >= 0.05)
    df["beats_frozen"] = (df.get("bf_boot_q", pd.Series(np.nan, index=df.index)) < 0.05) & (df.bf_ci_lo > 0)
    df["beats_e2e"] = (df.get("be_boot_q", pd.Series(np.nan, index=df.index)) < 0.05) & (df.be_ci_lo > 0)
    df.to_csv(out / "best_model_bootstrap.csv", index=False)
    print(f"wrote {out/'best_model_bootstrap.csv'}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()

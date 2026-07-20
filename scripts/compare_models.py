"""Paired model-vs-model comparison on scaffold k-fold CV — reusable in analysis + notebook.

For two runs evaluated on the SAME scaffold folds (same partition), report per task:
  - mean ± std across folds for each model,
  - fold-level paired t-test (n=folds; ANTI-CONSERVATIVE — CV folds share training data,
    Bengio & Grandvalet 2004 — so treat as indicative),
  - the RIGOROUS point test on the pooled out-of-fold predictions:
      * regression → paired Wilcoxon signed-rank on per-molecule squared error,
      * classification → DeLong paired-AUC test (fast, Sun & Xu 2014).
Classification tasks with multiple label columns (Tox21) are DeLong'd per column and summarised.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

# ---------------- fast DeLong (two correlated ROC curves) ----------------

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def delong_test(y_true, p1, p2):
    """Return (auc1, auc2, p) for H0: AUC1==AUC2 on the SAME samples (paired)."""
    y = np.asarray(y_true).astype(int)
    order = (-y).argsort(kind="mergesort")          # positives first
    label = y[order]; m = int(label.sum()); n = len(label) - m
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]]).astype(float)
    pos, neg = preds[:, :m], preds[:, m:]
    k = 2
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    s = sx / m + sy / n
    L = np.array([1.0, -1.0])
    var = L @ s @ L
    if var <= 0:
        return aucs[0], aucs[1], (1.0 if aucs[0] == aucs[1] else 0.0)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(z)))

# ---------------- data access ----------------

def _cv_csv(run): return f"figure_data/climb_v2_phase2/{run}/moleculenet_cv/moleculenet_summary.csv"
def _oof(run):    return f"figure_data/climb_v2_phase2/{run}/moleculenet_cv/test_predictions.csv"

def _folds(run, task, metric=None):
    """Per-fold values for one (task, metric). Classification summaries now carry BOTH
    `roc_auc` and `nef1` rows per fold, so metric MUST be specified to avoid mixing them."""
    d = pd.read_csv(_cv_csv(run))
    d = d[(d.dataset == task) & (d.head_seed.astype(str).str.startswith("fold"))]
    if metric is not None and "main_metric" in d.columns:
        d = d[d.main_metric == metric]
    d = d.sort_values("head_seed")
    return d.main_value.values

def _oof_task(run, task):
    d = pd.read_csv(_oof(run))
    return d[d.dataset == task].drop_duplicates(["dataset", "mol_index", "output_index"])

# ---------------- the comparison ----------------

def compare(run_a, run_b, tasks):
    """run_a vs run_b (b is usually the baseline, e.g. fp_desc). Returns a tidy DataFrame.

    `tasks` items are (task, higher_better) or (task, higher_better, metric). The metric
    selects which fold summary drives the mean±sd / fold-t columns: default 'rmse' for
    regression, 'roc_auc' for classification; pass 'nef1' for HIV-style early-enrichment.
    The rigorous point test is unchanged (Wilcoxon on sq-err for regression, DeLong paired
    AUC for classification — a rank test that tracks enrichment)."""
    rows = []
    for spec in tasks:
        task, higher_better = spec[0], spec[1]
        metric = spec[2] if len(spec) > 2 else ("roc_auc" if higher_better else "rmse")
        metric_label = {"rmse": "RMSE↓", "roc_auc": "AUC↑", "nef1": "NEF1%↑"}.get(metric, metric)
        try:
            a, b = _folds(run_a, task, metric), _folds(run_b, task, metric)
        except FileNotFoundError:
            a = b = np.array([])
        if len(a) == 0 or len(b) == 0:   # metric not evaluated yet (e.g. HIV NEF1% pre-final-pass)
            print(f"  [compare] skipping {task}/{metric}: no data for {run_a if len(a)==0 else run_b}")
            continue
        _, ptt = stats.ttest_rel(a, b)
        oa, ob = _oof_task(run_a, task), _oof_task(run_b, task)
        m = oa.merge(ob, on=["dataset", "mol_index", "output_index"], suffixes=("_a", "_b"))
        if higher_better:  # classification → DeLong (per label column, then summarise)
            ptests, favor_b = [], 0
            for oi, g in m.groupby("output_index"):
                gg = g[np.isfinite(g.y_true_a)]
                auc_a, auc_b, p = delong_test(gg.y_true_a, gg.y_pred_a, gg.y_pred_b)
                if np.isfinite(p):
                    ptests.append(p)
                    if auc_b > auc_a:
                        favor_b += 1
            point_p = float(np.median(ptests)) if ptests else np.nan
            point_test = f"DeLong (median p; {favor_b}/{len(ptests)} favour {run_b})" if len(ptests) > 1 else "DeLong"
        else:              # regression → paired Wilcoxon on per-molecule squared error
            ea = (m.y_pred_a - m.y_true_a) ** 2
            eb = (m.y_pred_b - m.y_true_b) ** 2
            _, point_p = stats.wilcoxon(ea, eb)
            point_test = "Wilcoxon (sq-err)"
        better_b = (b.mean() > a.mean()) if higher_better else (b.mean() < a.mean())
        rows.append(dict(task=task, metric=metric_label,
                         a_mean=a.mean(), a_sd=a.std(), b_mean=b.mean(), b_sd=b.std(),
                         delta=a.mean() - b.mean(), fold_t_p=ptt, point_test=point_test,
                         point_p=point_p, winner=(run_b if better_b else run_a)))
    return pd.DataFrame(rows)

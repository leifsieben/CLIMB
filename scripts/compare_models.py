"""Paired model-vs-model comparison on scaffold k-fold CV — reusable in analysis + notebook.

For two runs evaluated on the SAME scaffold folds (same partition), report per task:
  - mean ± std across folds for each model,
  - fold-level paired t-test (n=folds; ANTI-CONSERVATIVE — CV folds share training data,
    Bengio & Grandvalet 2004 — so treat as indicative),
  - an APPROXIMATE molecule-level point test on the pooled out-of-fold predictions:
      * regression → paired Wilcoxon signed-rank on per-molecule squared error,
      * classification → DeLong paired-AUC test (fast, Sun & Xu 2014).
    ⚠️ These treat molecules as independent. They are NOT: molecules that share a Bemis–Murcko
    scaffold are correlated, and the 5 folds' OOF predictions come from overlapping training sets,
    so the effective sample size is inflated and these p-values run anti-conservative (too small).
    They are indicative, not confirmatory — read the scaffold cluster-bootstrap CI (below) as the
    honest uncertainty.
  - a SCAFFOLD CLUSTER-BOOTSTRAP CI on the metric difference (opt-in, `n_boot>0`): resample whole
    Bemis–Murcko scaffolds (not molecules), recompute the paired metric difference each time, and
    report a percentile CI + a two-sided bootstrap p. This respects the clustering the tests above
    ignore, and it works for ANY metric — so HIV's NEF1% gets an honest interval, not just AUC.

Multiplicity: `bh_fdr()` gives Benjamini–Hochberg q-values; apply across the whole family of
(arm × task) comparisons (see `compare_many`). Classification tasks with multiple label columns
(Tox21) are scored per column and summarised.
"""
from __future__ import annotations
from functools import lru_cache
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

# ---------------- multiplicity + clustered uncertainty ----------------

def bh_fdr(pvals):
    """Benjamini–Hochberg FDR-adjusted q-values. NaNs pass through; result clipped to [0,1]."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return q
    ps = p[idx]; order = np.argsort(ps); n = ps.size
    adj = ps[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]          # enforce monotonic non-decreasing
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    q[idx] = out
    return q

@lru_cache(maxsize=300000)
def _scaffold(smiles):
    """Bemis–Murcko generic scaffold SMILES (falls back to the input on any RDKit failure)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or smiles
    except Exception:
        return smiles

def _auc(y, p):
    y = np.asarray(y); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def _nef1(y, p):
    y = np.asarray(y); N = len(y); A = int((y == 1).sum())
    if A == 0 or N == 0:
        return np.nan
    n = int(np.ceil(0.01 * N))
    top = np.argsort(-np.asarray(p, dtype=float))[:n]
    return int((y[top] == 1).sum()) / min(n, A)

def _metric_over_cols(sub, pred_col, kind):
    """Metric on a set of rows, averaged over output columns (Tox21=12), NaN-safe."""
    vals = []
    for _, g in sub.groupby("output_index"):
        yt = g["y_true_a"].to_numpy(); pr = g[pred_col].to_numpy(float)
        ok = np.isfinite(yt)
        if ok.sum() == 0:
            continue
        yt, pr = yt[ok], pr[ok]
        v = (np.sqrt(np.mean((pr - yt) ** 2)) if kind == "rmse"
             else _auc(yt, pr) if kind == "auc" else _nef1(yt, pr))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def cluster_bootstrap_diff(m, kind, higher_better, n_boot=1000, seed=0):
    """Scaffold cluster bootstrap of the paired metric difference (a − b, oriented so >0 = a better).

    Resamples whole Bemis–Murcko scaffolds with replacement (the correct unit given scaffold-mates
    are not independent), so the CI reflects clustered uncertainty. Returns obs diff, percentile CI,
    two-sided bootstrap p, and #scaffolds."""
    scaf = m["raw_smiles_a"].map(_scaffold).to_numpy()
    groups = {}
    for pos, s in enumerate(scaf):
        groups.setdefault(s, []).append(pos)
    keys = list(groups); K = len(keys)
    idx_of = {k: np.array(v) for k, v in groups.items()}
    rng = np.random.default_rng(seed)

    def diff_on(row_pos):
        sub = m.iloc[row_pos]
        da = _metric_over_cols(sub, "y_pred_a", kind)
        db = _metric_over_cols(sub, "y_pred_b", kind)
        if not (np.isfinite(da) and np.isfinite(db)):
            return np.nan
        return (da - db) if higher_better else (db - da)

    obs = diff_on(np.arange(len(m)))
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, K, K)
        rows = np.concatenate([idx_of[keys[i]] for i in pick])
        d = diff_on(rows)
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return dict(boot_diff=obs, ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan, n_scaffolds=K)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(boot_diff=obs, ci_lo=float(lo), ci_hi=float(hi), boot_p=float(min(p, 1.0)),
                n_scaffolds=K)

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

def compare(run_a, run_b, tasks, n_boot=0, boot_seed=0):
    """run_a vs run_b (b is usually the baseline, e.g. fp_desc). Returns a tidy DataFrame.

    `tasks` items are (task, higher_better) or (task, higher_better, metric). The metric
    selects which fold summary drives the mean±sd / fold-t columns: default 'rmse' for
    regression, 'roc_auc' for classification; pass 'nef1' for HIV-style early-enrichment.
    The molecule-level point test (Wilcoxon on sq-err / DeLong paired-AUC) is APPROXIMATE — it
    ignores scaffold clustering and fold overlap, so it is anti-conservative (see module docstring).

    `n_boot>0` adds the scaffold cluster-bootstrap CI columns (`ci_lo`, `ci_hi`, `boot_p`,
    `n_scaffolds`) on the *selected metric's* difference — this is the honest interval, and it
    covers NEF1% too. Default `n_boot=0` skips it so live notebook calls stay fast/unchanged."""
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
            point_test = f"DeLong~ (median p; {favor_b}/{len(ptests)} favour {run_b})" if len(ptests) > 1 else "DeLong~"
        else:              # regression → paired Wilcoxon on per-molecule squared error
            ea = (m.y_pred_a - m.y_true_a) ** 2
            eb = (m.y_pred_b - m.y_true_b) ** 2
            _, point_p = stats.wilcoxon(ea, eb)
            point_test = "Wilcoxon~ (sq-err)"          # ~ = molecule-level, anti-conservative
        better_b = (b.mean() > a.mean()) if higher_better else (b.mean() < a.mean())
        boot_kind = {"rmse": "rmse", "roc_auc": "auc", "nef1": "nef1"}.get(metric, "rmse")
        boot = (cluster_bootstrap_diff(m, boot_kind, higher_better, n_boot=n_boot, seed=boot_seed)
                if n_boot else dict(ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan, n_scaffolds=np.nan))
        rows.append(dict(task=task, metric=metric_label,
                         a_mean=a.mean(), a_sd=a.std(), b_mean=b.mean(), b_sd=b.std(),
                         delta=a.mean() - b.mean(), fold_t_p=ptt,
                         point_test=point_test, point_p=point_p,
                         ci_lo=boot["ci_lo"], ci_hi=boot["ci_hi"], boot_p=boot["boot_p"],
                         n_scaffolds=boot["n_scaffolds"], winner=(run_b if better_b else run_a)))
    return pd.DataFrame(rows)


def compare_many(pairs, tasks, n_boot=0, boot_seed=0):
    """Run compare() over many (run_a, run_b) pairs and FDR-correct across the whole family.

    Returns one tidy DataFrame with a `pair` column plus `point_q` and (if `n_boot>0`) `boot_q`:
    Benjamini–Hochberg q-values over every (arm × task) test at once, which is the multiplicity
    correction the comparison needs when many arms are judged over six tasks."""
    frames = []
    for ra, rb in pairs:
        d = compare(ra, rb, tasks, n_boot=n_boot, boot_seed=boot_seed)
        d.insert(0, "pair", f"{ra} vs {rb}")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["point_q"] = bh_fdr(out["point_p"].to_numpy())
        if n_boot:
            out["boot_q"] = bh_fdr(out["boot_p"].to_numpy())
    return out

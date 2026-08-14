"""Cross-task model comparison for the CheMeleon suite, Burns-style but AGGREGATED: a forest plot of each
model's MEAN RANK across tasks (1 = best on a task) with a bootstrap 95% CI over tasks. Rank is used
because Polaris mixes metrics (Pearson/Spearman/R2/ROC-AUC/PR-AUC) that cannot be averaged directly;
rank is direction-aware and scale-free, so it pools tasks honestly. Overlapping CIs => models that are
statistically indistinguishable across the suite (this is the point: it shows CheMeleon and
XGBoost(fp+desc) sitting together at the top rather than crowning a single winner).

Four panels: MoleculeACE overall / cliff-only / non-cliff / Polaris. Saves a PNG + a mean-rank CSV.
Run with .venv_sanity python (matplotlib, numpy)."""
import csv, json, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."
man = json.load(open("chemeleon_suite/data/polaris/polaris_manifest.json"))
HIGHER = {"pearsonr", "spearmanr", "r2", "roc_auc", "pr_auc", "accuracy", "explained_var"}
rng = np.random.default_rng(0)

# (display, source, dir/refname, color)
MODELS = [
    ("XGBoost (fp)", "ours", "ecfp4", "#E8820C"),
    ("XGBoost (fp+desc)", "ours", "fp_desc", "#B5651D"),
    ("CheMeleon (e2e)", "ref", "CheMeleon", "#1f77b4"),
    ("unsup_8M (frozen)", "ours", "unsup_8M", "#9ecae1"),
    ("unsup_8M (e2e)", "ours", "unsup_8M_e2e", "#3182bd"),
    ("sup_dense_8M (frozen)", "ours", "skip_dense_8M", "#a1d99b"),
    ("sup_dense_8M (e2e)", "ours", "skip_dense_8M_e2e", "#31a354"),
    ("sup_sparse_8M", "ours", "skip_sparse_all_8M", "#c7c7c7"),
    ("sup_dense+sparse_8M", "ours", "skip_dense_plus_sparse_8M", "#969696"),
    ("no_pretrain_e2e", "ours", "no_pretrain_e2e_e2e", "#d62728"),
    ("no_pretrain_random", "ours", "random_baseline_00", "#fb9a99"),
]
RAW = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("chemeleon_suite/reference/reference_long.csv")):
    RAW[(r["track"], r["model"])][(r["task"], r["metric"])].append(float(r["value"]))

def ours_mace(d, subset):
    by = defaultdict(list)
    try:
        for r in csv.DictReader(open(f"figure_data/chemeleon_suite/moleculeace/{d}/results.csv")):
            if r["subset"] == subset and r["metric"] == "rmse": by[r["task"]].append(float(r["value"]))
    except FileNotFoundError: return {}
    return {t: st.mean(v) for t, v in by.items()}
def ref_mace(name, subset):
    me = {"overall": "overall test rmse", "cliff": "cliff test rmse", "noncliff": "noncliff test rmse"}[subset]
    return {t: st.mean(v) for (t, m), v in RAW[("moleculeace", name)].items() if m == me}
def ours_pol(d):
    by = defaultdict(lambda: defaultdict(list))
    try:
        for r in csv.DictReader(open(f"figure_data/chemeleon_suite/polaris/{d}/polaris_scores.csv")): by[r["task"]][r["metric"]].append(float(r["value"]))
    except FileNotFoundError: return {}
    return {t: st.mean(m[man[t]["primary_metric"]]) for t, m in by.items() if man.get(t, {}).get("primary_metric") in m}
def ref_pol(name):
    out = {}
    for (t, m), v in RAW[("polaris", name)].items():
        if m == man.get(t, {}).get("primary_metric"): out[t] = st.mean(v)
    return out

def category_vals(cat):
    vals = {}
    for disp, src, name, _ in MODELS:
        if cat == "polaris":
            vals[disp] = ours_pol(name) if src == "ours" else ref_pol(name)
        else:
            sub = {"mace_overall": "overall", "mace_cliff": "cliff", "mace_noncliff": "noncliff"}[cat]
            vals[disp] = ours_mace(name, sub) if src == "ours" else ref_mace(name, sub)
    return vals

def rank_stats(cat):
    vals = category_vals(cat)
    labels = [m[0] for m in MODELS]
    tasks = sorted(set.intersection(*[set(vals[l]) for l in labels]))
    higher = (cat == "polaris")
    # per-task rank matrix [ntask x nmodel]
    R = np.zeros((len(tasks), len(labels)))
    for i, t in enumerate(tasks):
        hi = higher and man[t]["primary_metric"] in HIGHER
        order = sorted(range(len(labels)), key=lambda j: vals[labels[j]][t], reverse=hi)
        for rk, j in enumerate(order, 1): R[i, j] = rk
    mean = R.mean(0)
    # bootstrap over tasks
    B = 3000; boot = np.zeros((B, len(labels)))
    for b in range(B):
        idx = rng.integers(0, len(tasks), len(tasks)); boot[b] = R[idx].mean(0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return labels, mean, lo, hi, len(tasks)

PANELS = [("mace_overall", "MoleculeACE — overall RMSE"), ("mace_cliff", "MoleculeACE — activity-cliff only"),
          ("mace_noncliff", "MoleculeACE — non-cliff"), ("polaris", "Polaris / TDC (28 tasks)")]
fig, axes = plt.subplots(2, 2, figsize=(13, 11)); axes = axes.ravel()
csvrows = []
for ax, (cat, title) in zip(axes, PANELS):
    labels, mean, lo, hi, nt = rank_stats(cat)
    order = np.argsort(mean)  # best (lowest rank) first
    y = np.arange(len(labels))[::-1]  # best at TOP
    cols = {m[0]: m[3] for m in MODELS}
    for yi, j in zip(y, order):
        ax.errorbar(mean[j], yi, xerr=[[mean[j]-lo[j]], [hi[j]-mean[j]]], fmt="o", color=cols[labels[j]],
                    ecolor=cols[labels[j]], capsize=3, ms=8, lw=2)
        csvrows.append({"panel": cat, "model": labels[j], "mean_rank": round(mean[j],3),
                        "ci_lo": round(lo[j],3), "ci_hi": round(hi[j],3), "n_tasks": nt})
    ax.set_yticks(y); ax.set_yticklabels([labels[j] for j in order], fontsize=9)
    ax.set_xlabel("mean rank across tasks (1 = best; ← better)")
    ax.set_title(f"{title}  ·  n={nt}", fontsize=11)
    ax.axvline(mean[order[0]], color="#bbb", ls="--", lw=1, zorder=0)  # best model's mean rank (reference)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("CheMeleon suite — cross-task model comparison (mean rank ± bootstrap 95% CI)\n"
             "overlapping CIs = statistically indistinguishable across the suite", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "chemeleon_suite/summaries/model_comparison_rank.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
with open("chemeleon_suite/summaries/model_comparison_rank.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["panel","model","mean_rank","ci_lo","ci_hi","n_tasks"]); w.writeheader(); w.writerows(csvrows)
print(f"wrote {out} + model_comparison_rank.csv")

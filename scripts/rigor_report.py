"""Reviewer-rigor report for the principal 8M comparisons.

Addresses three review points, all from data that already exists (no retraining):

  (1) Pretraining-seed variance: for the four principal 8M arms we have 3 independent pretraining
      seeds (s0/s1/s2), each fully CV'd. Report mean +/- std ACROSS seeds per task, alongside the
      fold std that the headline error bars currently use — so seed variability is visible.
  (2) Multiplicity: Benjamini-Hochberg FDR q-values across the whole family of (pair x task) tests.
  (3) Clustered uncertainty: scaffold cluster-bootstrap CI on each metric difference (resample whole
      Bemis-Murcko scaffolds, not molecules), which is honest about scaffold non-independence and
      also gives NEF1% its own interval.

Writes analysis/rigor/{seed_variance,comparison_fdr_bootstrap}.{csv,md}.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import scripts.compare_models as C

OUT = Path("analysis/rigor"); OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000


def _md_table(df):
    """Minimal DataFrame -> GitHub markdown table (avoids the optional `tabulate` dependency)."""
    df = df.round(4)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n"
                   for row in df.itertuples(index=False, name=None))
    return head + body

# principal 8M arms (pretty name -> base run) that have 3 pretraining seeds
ARMS = {"unsup_only": "unsup_8M", "sup_only:dense": "skip_dense_8M",
        "sup_only:mixed": "skip_mixed_8M", "unsup->sup:dense": "u2s_dense_from8M"}
TASKS = [("ESOL", False), ("QM7", False), ("BBBP", True), ("BACE", True),
         ("Tox21", True), ("HIV", True), ("HIV", True, "nef1")]
# headline pairwise claims (a vs b)
PAIRS = [("unsup_8M", "fp_desc_anchor"), ("unsup_8M", "skip_dense_8M"),
         ("u2s_dense_from8M", "skip_dense_8M"), ("u2s_dense_from8M", "unsup_8M")]


def seed_variance():
    def mean_for(run, ds, metric, stat):
        d = pd.read_csv(C._cv_csv(run))
        r = d[(d.dataset == ds) & (d.main_metric == metric) & (d.head_seed == stat)]
        return float(r.main_value.iloc[0]) if len(r) else np.nan
    rows = []
    for arm, base in ARMS.items():
        for spec in TASKS:
            ds = spec[0]; metric = spec[2] if len(spec) > 2 else ("roc_auc" if spec[1] else "rmse")
            runs = [base] + [f"{base}_s{s}" for s in (1, 2)]
            seed_means = [mean_for(r, ds, metric, "MEAN") for r in runs]
            fold_std_s0 = mean_for(base, ds, metric, "STD")
            if any(np.isnan(seed_means)):
                continue
            rows.append(dict(arm=arm, task=ds, metric=metric,
                             seed_mean=np.mean(seed_means), seed_std=np.std(seed_means),
                             fold_std_s0=fold_std_s0,
                             s0=seed_means[0], s1=seed_means[1], s2=seed_means[2]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_variance.csv", index=False)
    with open(OUT / "seed_variance.md", "w") as fh:
        fh.write("# Pretraining-seed variance, principal 8M arms (3 seeds x 5-fold CV)\n\n")
        fh.write("`seed_std` = std across the 3 pretraining seeds' fold-means; `fold_std_s0` = the "
                 "within-seed across-fold std currently used for the headline error bars.\n\n")
        fh.write(_md_table(df))
    print("seed_variance: wrote", len(df), "rows")
    return df


def comparisons():
    df = C.compare_many(PAIRS, TASKS, n_boot=N_BOOT, boot_seed=0)
    keep = ["pair", "task", "metric", "a_mean", "b_mean", "delta",
            "fold_t_p", "point_p", "point_q", "ci_lo", "ci_hi", "boot_p", "boot_q", "n_scaffolds"]
    df = df[keep]
    df.to_csv(OUT / "comparison_fdr_bootstrap.csv", index=False)
    with open(OUT / "comparison_fdr_bootstrap.md", "w") as fh:
        fh.write("# Principal 8M comparisons — FDR-corrected + scaffold cluster-bootstrap\n\n")
        fh.write(f"Cluster bootstrap: {N_BOOT} resamples of whole Bemis-Murcko scaffolds. "
                 "`point_p` = molecule-level DeLong/Wilcoxon (anti-conservative, indicative only); "
                 "`point_q`/`boot_q` = Benjamini-Hochberg FDR across all rows; "
                 "`[ci_lo, ci_hi]` = 95% cluster-bootstrap CI on the metric difference (a - b, "
                 "oriented so >0 favours a). A CI spanning 0 = no difference detectable.\n\n")
        fh.write(_md_table(df))
    print("comparisons: wrote", len(df), "rows")
    return df


if __name__ == "__main__":
    sv = seed_variance()
    cm = comparisons()
    # headline takeaways
    big_seed = sv[sv.seed_std >= 0.01]
    print(f"\nTasks where pretraining-seed std >= 0.01 (seed noise is material): {len(big_seed)}")
    for _, r in big_seed.iterrows():
        print(f"  {r.arm:16s} {r.task:6s} seed_std={r.seed_std:.3f} (fold_std={r.fold_std_s0:.3f})")
    flipped = cm[(cm.point_p < 0.05) & (cm.boot_p >= 0.05)]
    print(f"\nRows significant molecule-level (p<.05) but NOT under the cluster bootstrap: {len(flipped)}")
    for _, r in flipped.iterrows():
        print(f"  {r.pair} | {r.task}/{r.metric}: point_p={r.point_p:.3g} boot_p={r.boot_p:.3g}")

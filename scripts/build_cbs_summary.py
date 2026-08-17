"""Aggregate the cbs virtual-screening battery into plottable CSVs (NEF1% headline, ROC-AUC
secondary), for a NEW MAINLINE figure. Reads each run's moleculenet_cv/suite_summary.json under
figure_data/cbs_benchmark/, maps run -> A1a arm, and reports per-arm mean +- std over pretraining
seeds (each per-run value is already the mean over the 5 PROVIDED folds; within-run STD = fold
spread). Emits:
  experiment_cbs/cbs_nef1_summary.csv   # per (arm, metric): mean, std_over_seeds, n_seeds  (THE figure input)
  experiment_cbs/cbs_per_run.csv        # per run: nef1 mean/std(fold), roc mean/std(fold)

Reference to overlay in the figure: Truong et al. 2026 SOTA generic VS pipelines NEF1%=0.764+-0.191.
Run from repo root with any python (stdlib only)."""
import csv
import json
import os
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = Path(ROOT) / "figure_data" / "cbs_benchmark"
OUT = Path(ROOT) / "experiment_cbs"
OUT.mkdir(parents=True, exist_ok=True)

# run -> (arm label, arm order). Mirrors notebook_cells/08.py A1_ORDER.
ARMS = [
    ("ecfp4",                       "ecfp4_anchor"),
    ("fp_desc",                     "fp_desc_anchor"),
    ("no_pretrain",                 "random_baseline_00,random_baseline_01,random_baseline_02"),
    ("no_pretrain_e2e",             "e2e_random_00,e2e_random_01,e2e_random_02"),
    ("unsup_only",                  "unsup_8M,unsup_8M_s1,unsup_8M_s2"),
    ("sup_only:dense",              "skip_dense_8M,skip_dense_8M_s1,skip_dense_8M_s2"),
    ("sup_only:sparse_all",         "skip_sparse_all_8M,skip_sparse_all_8M_s1,skip_sparse_all_8M_s2"),
    ("sup_only:dense_plus_sparse",  "skip_dense_plus_sparse_8M,skip_dense_plus_sparse_8M_s1,skip_dense_plus_sparse_8M_s2"),
    ("unsup2sup:dense",             "u2s_dense_from8M,u2s_dense_from8M_s1,u2s_dense_from8M_s2"),
    ("unsup2sup:sparse_all",        "u2s_sparse_all_from8M,u2s_sparse_all_from8M_s1,u2s_sparse_all_from8M_s2"),
    ("unsup2sup:dense_plus_sparse", "u2s_dense_plus_sparse_from8M,u2s_dense_plus_sparse_from8M_s1,u2s_dense_plus_sparse_from8M_s2"),
    # The four long-standing CBS gaps, filled 2026-08-17. NOTE: base pretraining seed only (the
    # _s1/_s2 replicates were never run on CBS for these recipes), so n_seeds=1 here vs 3 above.
    ("sup_only:mixed",              "skip_mixed_8M"),
    ("sup_only:minimol_full",       "skip_minimol_full_8M"),
    ("unsup2sup:mixed",             "u2s_mixed_from8M"),
    ("unsup2sup:minimol_full",      "u2s_minimol_full_from8M"),
    # Catastrophic-forgetting mirror (sup 8M -> 2M MLM). Picked up automatically once it lands.
    ("sup2unsup:dense",             "s2u_dense_from8M_s0,s2u_dense_from8M_s1,s2u_dense_from8M_s2"),
    # External CheMeleon / chemprop comparators (Burns et al. 2025), same provided folds + NEF1%:
    ("chemeleon_frozen",            "chemeleon_frozen"),                                       # CheMeleon fingerprint + probe
    ("chemprop_e2e",                "chemprop_e2e_s0,chemprop_e2e_s1,chemprop_e2e_s2"),        # vanilla D-MPNN, e2e
    ("chemeleon_e2e",               "chemeleon_e2e_s0,chemeleon_e2e_s1,chemeleon_e2e_s2"),     # CheMeleon foundation, e2e
]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    """SAMPLE sd (ddof=1) over pretraining seeds — these are a sample of training runs, not the
    population. Was ddof=0 until 2026-08-17, which understated the spread by sqrt(n/(n-1)) = 22%
    at n=3 and disagreed with the sample sd the figure layer uses."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _load(run):
    p = FD / run / "moleculenet_cv" / "suite_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    per_run_rows, summ_rows = [], []
    for arm, runs_csv in ARMS:
        runs = runs_csv.split(",")
        nef_means, roc_means = [], []
        for run in runs:
            d = _load(run)
            if d is None:
                print(f"[cbs-summary] MISSING {run} ({arm})")
                continue
            nm, ns = d.get("cbs_nef1_MEAN"), d.get("cbs_nef1_STD")
            rm, rs = d.get("cbs_MEAN"), d.get("cbs_STD")
            if nm is not None:
                nef_means.append(nm)
            if rm is not None:
                roc_means.append(rm)
            per_run_rows.append({"arm": arm, "run": run, "nef1_mean": nm, "nef1_std_fold": ns,
                                 "roc_auc_mean": rm, "roc_auc_std_fold": rs})
        summ_rows.append({"arm": arm, "metric": "nef1", "mean": _mean(nef_means),
                          "std_over_seeds": _std(nef_means), "n_seeds": len(nef_means)})
        summ_rows.append({"arm": arm, "metric": "roc_auc", "mean": _mean(roc_means),
                          "std_over_seeds": _std(roc_means), "n_seeds": len(roc_means)})

    with (OUT / "cbs_per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "run", "nef1_mean", "nef1_std_fold",
                                          "roc_auc_mean", "roc_auc_std_fold"])
        w.writeheader(); w.writerows(per_run_rows)
    with (OUT / "cbs_nef1_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "metric", "mean", "std_over_seeds", "n_seeds"])
        w.writeheader(); w.writerows(summ_rows)

    # Correctly-labelled reference lines to overlay (prevents the "0.764=generic" mislabel from
    # propagating into a caption). 0.764 is Truong's CBS-SPECIFIC target-trained structure-based
    # models; generic docking/co-folding pipelines ~0; their ligand-only descriptor baseline 0.000-0.125.
    ref_rows = [
        {"label": "Truong CBS-specific (target-trained, structure-based)", "metric": "nef1",
         "value": 0.764, "err": 0.191, "source": "Truong 2026 Fig 4A / abstract"},
        {"label": "SOTA generic (docking/co-folding, 16 pipelines)", "metric": "nef1",
         "value": 0.0, "err": "", "source": "Truong 2026 Fig 3A / Table S2 (all 16 = 0.000 on Fold 3); refine from Data S1"},
        {"label": "Truong ligand-only (RDKit descriptors + logreg)", "metric": "nef1",
         "value": 0.0625, "err": 0.0625, "source": "Truong 2026 text: NEF1% 0.000-0.125 (range midpoint)"},
        {"label": "OUR reproduction of ligand-only (rdkit_desc + linear, provided folds)", "metric": "nef1",
         "value": 0.070, "err": 0.058, "source": "this work — artifact control"},
    ]
    with (OUT / "cbs_reference_lines.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "metric", "value", "err", "source"])
        w.writeheader(); w.writerows(ref_rows)

    print("\n[cbs-summary] NEF1% by arm (mean +- std over seeds; Truong CBS-specific target-trained = 0.764 +- 0.191; generic SOTA ~0):")
    for r in summ_rows:
        if r["metric"] == "nef1":
            print(f"   {r['arm']:<30} {r['mean']:.3f} +- {r['std_over_seeds']:.3f}  (n_seeds={r['n_seeds']})")
    print(f"\n[cbs-summary] wrote {OUT/'cbs_nef1_summary.csv'} and {OUT/'cbs_per_run.csv'}")


if __name__ == "__main__":
    main()

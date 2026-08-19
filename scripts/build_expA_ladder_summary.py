"""Experiment A — stitch the synthetic-statistics ladder into one tidy CSV for the notebook session.

Rungs (each = the SAME frozen-probe eval, 5-fold scaffold CV, 3 pretraining seeds):
  real (unsup_only)   : unsup_8M{,_s1,_s2}                 [climb_v2_phase2]
  shuffle_tokens      : corrupt_mlm_8M(+_s1,_s2)           [phase2 s0 + climb_v2_expA s1/s2]
  unigram_resample    : unigram_8M{,_s1,_s2}               [climb_v2_expA]           <- NEW
  no_pretrain (frozen): random_baseline_0{0,1,2}           [phase2]
  no_pretrain (e2e)   : e2e_random_0{0,1,2}                [phase2]  (end-to-end, different protocol)

For each run we read moleculenet_cv/moleculenet_summary.csv and take the per-fold aggregate rows
(main_metric == primary, head_seed like 'fold0'..'fold4'); the run's CV score per task = mean over
its 5 folds. Then we report mean±std across the 3 pretraining seeds per (rung, task).

Reads from S3 (results are backed up there). Writes:
  analysis/rigor/expA_ladder_per_run.csv     (arm,run_id,seed,wave,dataset,task_type,metric,cv_value)
  analysis/rigor/expA_ladder_summary.csv     (arm,dataset,task_type,metric,mean,std,n_seeds)

Usage: python scripts/build_expA_ladder_summary.py
"""
from __future__ import annotations

import io
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "s3://climb-s3-bucket/experiments"
FOLD_RE = re.compile(r"^fold\d+$")

# (arm label, run_id, pretraining_seed, wave)
# UNITS: the frozen comparators MUST come from the native-unit re-eval under
# climb_v2_expA/_baselines (scripts/expA_baselines_native_eval.sh), NOT the phase2 moleculenet_cv,
# which is in normalized units for regression (QM7 rmse ~0.87 vs native ~200). Mixing the two
# silently corrupts the regression ladder. unigram/shuffle_s1s2/bigram are already native (expA wave).
BASE = "climb_v2_expA/_baselines"   # native re-eval of the phase2 frozen comparators
ARMS = [
    ("real (unsup_only)", "unsup_8M", 0, BASE),
    ("real (unsup_only)", "unsup_8M_s1", 1, BASE),
    ("real (unsup_only)", "unsup_8M_s2", 2, BASE),
    ("shuffle_tokens", "corrupt_mlm_8M", 0, BASE),
    ("shuffle_tokens", "corrupt_mlm_8M_s1", 1, "climb_v2_expA"),
    ("shuffle_tokens", "corrupt_mlm_8M_s2", 2, "climb_v2_expA"),
    ("unigram_resample", "unigram_8M", 0, "climb_v2_expA"),
    ("unigram_resample", "unigram_8M_s1", 1, "climb_v2_expA"),
    ("unigram_resample", "unigram_8M_s2", 2, "climb_v2_expA"),
    ("bigram_resample", "bigram_8M", 0, "climb_v2_expA"),
    ("bigram_resample", "bigram_8M_s1", 1, "climb_v2_expA"),
    ("bigram_resample", "bigram_8M_s2", 2, "climb_v2_expA"),
    ("no_pretrain (frozen)", "random_baseline_00", 0, BASE),
    ("no_pretrain (frozen)", "random_baseline_01", 1, BASE),
    ("no_pretrain (frozen)", "random_baseline_02", 2, BASE),
    # e2e is end-to-end (different protocol) and its phase2 regression is NORMALIZED units — keep for
    # classification context only; do not read its regression as native-comparable.
    ("no_pretrain (e2e)", "e2e_random_00", 0, "climb_v2_phase2"),
    ("no_pretrain (e2e)", "e2e_random_01", 1, "climb_v2_phase2"),
    ("no_pretrain (e2e)", "e2e_random_02", 2, "climb_v2_phase2"),
]

ARM_ORDER = ["real (unsup_only)", "shuffle_tokens", "bigram_resample", "unigram_resample",
             "no_pretrain (frozen)", "no_pretrain (e2e)"]


# HIV took CBS's rare-active-screen slot in the canonical six on 2026-08-19 and is scored on NEF1%,
# which is NOT the dataset's primary metric -- so the ladder has to emit it explicitly or fig_E's
# unsupervised panel silently plots HIV ROC-AUC against a NEF1% floor.
EXTRA_METRICS = {"HIV": ["nef1"]}


def _read_cv(wave: str, run_id: str) -> pd.DataFrame | None:
    # prefer a locally synced copy; fall back to S3
    local = Path(__file__).resolve().parent.parent / "figure_data" / wave / run_id / \
        "moleculenet_cv" / "moleculenet_summary.csv"
    if local.exists():
        return pd.read_csv(local)
    uri = f"{BUCKET}/{wave}/{run_id}/moleculenet_cv/moleculenet_summary.csv"
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True)
    if r.returncode != 0:
        print(f"  MISSING {run_id}: {r.stderr.decode()[:120]}")
        return None
    return pd.read_csv(io.StringIO(r.stdout.decode()))


def main() -> int:
    rows = []
    for arm, run_id, seed, wave in ARMS:
        df = _read_cv(wave, run_id)
        if df is None:
            continue
        # per-fold aggregate rows only (the ones the notebook consumes)
        fold = df[df.head_seed.astype(str).map(lambda s: bool(FOLD_RE.match(s)))]
        for ds, g in fold.groupby("dataset"):
            ttype = g.task_type.iloc[0]
            # e2e regression in phase2 is NORMALIZED units (QM7 ~0.85) and end-to-end (different
            # protocol) — not comparable on the native frozen-probe axis. Keep e2e for classification
            # only, so the regression ladder stays unit-consistent.
            if arm == "no_pretrain (e2e)" and ttype == "regression":
                continue
            primary = "rmse" if ttype == "regression" else "roc_auc"
            for metric in [primary] + EXTRA_METRICS.get(ds, []):
                gg = g[g.main_metric == metric]
                if gg.empty:
                    continue
                rows.append({
                    "arm": arm, "run_id": run_id, "seed": seed, "wave": wave,
                    "dataset": ds, "task_type": ttype, "metric": metric,
                    "cv_value": float(gg.main_value.mean()),  # mean over the 5 folds
                    "n_folds": int(len(gg)),
                })

    per_run = pd.DataFrame(rows)
    out = Path("analysis/rigor"); out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out / "expA_ladder_per_run.csv", index=False)

    summ = (per_run.groupby(["arm", "dataset", "task_type", "metric"], as_index=False)
            .agg(mean=("cv_value", "mean"), std=("cv_value", "std"), n_seeds=("cv_value", "size")))
    summ.to_csv(out / "expA_ladder_summary.csv", index=False)

    # ---- printed sanity: ladder per task ----
    print(f"\nwrote analysis/rigor/expA_ladder_per_run.csv ({len(per_run)} rows) and expA_ladder_summary.csv")
    for ds in sorted(per_run.dataset.unique()):
        # a dataset can now carry MORE THAN ONE metric (HIV: roc_auc + nef1), so loop over them --
        # indexing by arm alone returned a Series per cell and crashed the format string.
        for metric, dm in summ[summ.dataset == ds].groupby("metric"):
            better = "lower" if metric == "rmse" else "higher"
            print(f"\n=== {ds} ({metric}, {better}=better) ===")
            dm = dm.set_index("arm")
            for arm in ARM_ORDER:
                if arm in dm.index:
                    r = dm.loc[arm]
                    print(f"  {arm:24} {r['mean']:.4f} ± {r['std']:.4f}  (n={int(r['n_seeds'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

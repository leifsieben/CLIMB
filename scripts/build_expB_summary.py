"""Experiment B — stitch the wiki-transfer result into one tidy CSV: wiki_real vs real vs no_pretrain,
all frozen-probe 5-fold CV in NATIVE units, same eval version. Comparators reuse the Exp A native
re-evals (climb_v2_expA/_baselines), so the 3 arms are directly comparable.

Writes analysis/rigor/expB_wiki_{per_run,summary}.csv. Reads per-run moleculenet_cv from S3.

Usage:  python scripts/build_expB_summary.py
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

# (arm, run_id, seed, wave)
ARMS = [
    ("wiki_real", "wiki_real_8M", 0, "climb_v2_expB"),
    ("wiki_real", "wiki_real_8M_s1", 1, "climb_v2_expB"),
    ("wiki_real", "wiki_real_8M_s2", 2, "climb_v2_expB"),
    ("real (unsup_only)", "unsup_8M", 0, "climb_v2_expA/_baselines"),
    ("real (unsup_only)", "unsup_8M_s1", 1, "climb_v2_expA/_baselines"),
    ("real (unsup_only)", "unsup_8M_s2", 2, "climb_v2_expA/_baselines"),
    ("no_pretrain (frozen)", "random_baseline_00", 0, "climb_v2_expA/_baselines"),
    ("no_pretrain (frozen)", "random_baseline_01", 1, "climb_v2_expA/_baselines"),
    ("no_pretrain (frozen)", "random_baseline_02", 2, "climb_v2_expA/_baselines"),
]
ARM_ORDER = ["real (unsup_only)", "wiki_real", "no_pretrain (frozen)"]


def _read_cv(wave, run_id):
    uri = f"{BUCKET}/{wave}/{run_id}/moleculenet_cv/moleculenet_summary.csv"
    r = subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True)
    if r.returncode != 0:
        print(f"  MISSING {wave}/{run_id}")
        return None
    return pd.read_csv(io.StringIO(r.stdout.decode()))


def main() -> int:
    rows = []
    for arm, run_id, seed, wave in ARMS:
        df = _read_cv(wave, run_id)
        if df is None:
            continue
        fold = df[df.head_seed.astype(str).map(lambda s: bool(FOLD_RE.match(s)))]
        for ds, g in fold.groupby("dataset"):
            ttype = g.task_type.iloc[0]
            primary = "rmse" if ttype == "regression" else "roc_auc"
            gg = g[g.main_metric == primary]
            if gg.empty:
                continue
            rows.append({"arm": arm, "run_id": run_id, "seed": seed, "dataset": ds,
                         "task_type": ttype, "metric": primary, "cv_value": float(gg.main_value.mean())})
    per = pd.DataFrame(rows)
    out = Path("analysis/rigor"); out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "expB_wiki_per_run.csv", index=False)
    summ = (per.groupby(["arm", "dataset", "task_type", "metric"], as_index=False)
            .agg(mean=("cv_value", "mean"), std=("cv_value", "std"), n_seeds=("cv_value", "size")))
    summ.to_csv(out / "expB_wiki_summary.csv", index=False)
    print(f"\nwrote analysis/rigor/expB_wiki_{{per_run,summary}}.csv ({len(per)} rows)")
    for ds in sorted(per.dataset.unique()):
        d = summ[summ.dataset == ds]; metric = d.metric.iloc[0]
        better = "lower" if metric == "rmse" else "higher"
        print(f"\n=== {ds} ({metric}, {better}=better) ===")
        d = d.set_index("arm")
        for arm in ARM_ORDER:
            if arm in d.index:
                r = d.loc[arm]
                print(f"  {arm:22} {r['mean']:.4f} ± {r['std']:.4f}  (n={int(r['n_seeds'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FOLD_RE = re.compile(r"^fold\d+$")


BASE = "climb_v2_expA/_baselines"
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


    ("no_pretrain (e2e)", "e2e_random_00", 0, "climb_v2_phase2"),
    ("no_pretrain (e2e)", "e2e_random_01", 1, "climb_v2_phase2"),
    ("no_pretrain (e2e)", "e2e_random_02", 2, "climb_v2_phase2"),
]

ARM_ORDER = ["real (unsup_only)", "shuffle_tokens", "bigram_resample", "unigram_resample",
             "no_pretrain (frozen)", "no_pretrain (e2e)"]


EXTRA_METRICS = {"HIV": ["nef1"]}


def _read_cv(wave, run_id):
    path = Path(__file__).resolve().parent.parent / "figure_data" / wave / run_id / "moleculenet_cv" / "moleculenet_summary.csv"
    if not path.exists():
        print(f"MISSING {path}")
        return None
    return pd.read_csv(path)


def main() -> int:
    rows = []
    for arm, run_id, seed, wave in ARMS:
        df = _read_cv(wave, run_id)
        if df is None:
            continue

        fold = df[df.head_seed.astype(str).map(lambda s: bool(FOLD_RE.match(s)))]
        for ds, g in fold.groupby("dataset"):
            ttype = g.task_type.iloc[0]


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
                    "cv_value": float(gg.main_value.mean()),
                    "n_folds": int(len(gg)),
                })

    per_run = pd.DataFrame(rows)
    out = Path("analysis/rigor"); out.mkdir(parents=True, exist_ok=True)

    summ = (per_run.groupby(["arm", "dataset", "task_type", "metric"], as_index=False)
            .agg(mean=("cv_value", "mean"), std=("cv_value", "std"), n_seeds=("cv_value", "size")))
    summ.to_csv(out / "expA_ladder_summary.csv", index=False)


    print(f"\nwrote analysis/rigor/expA_ladder_summary.csv")
    for ds in sorted(per_run.dataset.unique()):


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

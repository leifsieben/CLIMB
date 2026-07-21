"""Workstream B — the eval-ceiling test: frozen probe vs fine-tuning.

The scaling "saturation" is suspected to be a low-ceiling eval artifact (frozen
mean-pool + small head can't resolve encoder quality — MLM loss varied 0.14→0.39
across runs but downstream stayed flat). This driver takes a set of encoders (e.g.
the compute-scaling runs) and, per task, reports BOTH the frozen-featurizer metric
(read from the launcher's moleculenet_summary.csv) and the end-to-end fine-tuned
metric (finetune_v2). If fine-tuning separates encoders the frozen probe couldn't,
the probe was the ceiling.

Usage:
    python scripts/run_eval_ceiling.py --results_root experiments/climb_v2_compute \
        --run_ids cscale_2M cscale_8M cscale_24M random_baseline_00 \
        --tasks BBBP BACE ESOL --output_dir experiments/climb_v2_compute/ceiling
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune_v2 import TASK_TYPE, finetune_one


def _budget(run_id: str) -> float:
    """Pretraining forward passes parsed from the run id.

    Was anchored to the old `cscale_<n>[MB]` naming, so every phase-2 id (unsup_2M,
    skip_dense_8M, ...) parsed as 0.0 -- and the per-task plots filter on `budget > 0`, so the
    driver would have written a CSV and then silently produced no figures at all.
    """
    m = re.search(r"(?:^|_)(\d+)([MB])(?:$|_)", run_id)
    if not m:
        return 0.0  # random baseline / anchor: no pretraining budget
    return int(m.group(1)) * (1e9 if m.group(2) == "B" else 1e6)


def _frozen_metric(results_root: Path, run_id: str, task: str):
    f = results_root / run_id / "moleculenet" / "moleculenet_summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    row = df[(df.dataset == task) & (df.head_seed == "MEAN")]
    return float(row["main_value"].iloc[0]) if len(row) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", required=True)
    p.add_argument("--run_ids", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=["BBBP", "BACE", "ESOL"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=None,
                   help="shared tokenizer dir; used when a run has no tokenizer/ of its own "
                        "(phase-2 runs share one tokenizer rather than copying it per run)")
    args = p.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id in args.run_ids:
        enc = root / run_id / "encoder"
        tok = root / run_id / "tokenizer"
        if not tok.exists() and args.tokenizer:
            tok = Path(args.tokenizer)
        for task in args.tasks:
            tt = TASK_TYPE.get(task, "classification")
            metric_name = "roc_auc" if tt == "classification" else "rmse"
            frozen = _frozen_metric(root, run_id, task)
            ft = None
            if enc.exists() and tok.exists():
                try:
                    ft = finetune_one(str(enc), str(tok), task, tt, seed=args.seed)
                except Exception as e:
                    print(f"[ceiling] finetune {run_id}/{task} failed: {e}")
            print(f"[ceiling] {run_id:22} {task:6} frozen={frozen} finetune={ft}")
            rows.append({"run_id": run_id, "budget": _budget(run_id), "task": task,
                         "metric": metric_name, "frozen": frozen, "finetune": ft})

    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_ceiling.csv", index=False)

    # per-task: metric vs compute budget, frozen line vs finetune line
    for task in sorted(df.task.unique()):
        d = df[(df.task == task) & (df.budget > 0)].sort_values("budget")
        if d.empty:
            continue
        metric = d["metric"].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        if d["frozen"].notna().any():
            ax.plot(d["budget"], d["frozen"], marker="o", label="frozen probe")
        if d["finetune"].notna().any():
            ax.plot(d["budget"], d["finetune"], marker="s", label="fine-tuned")
        rb = df[(df.task == task) & (df.run_id.str.startswith("random"))]
        if not rb.empty and rb["frozen"].notna().any():
            ax.axhline(rb["frozen"].mean(), ls="--", color="#999", label="random (frozen)")
        ax.set_xscale("log")
        ax.set_xlabel("pretraining forward passes")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{task} — eval ceiling: frozen vs fine-tuned ({metric})")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"fig_ceiling_{task}.png", dpi=120); plt.close(fig)
    print(f"[ceiling] wrote {out}/eval_ceiling.csv + figures")


if __name__ == "__main__":
    main()

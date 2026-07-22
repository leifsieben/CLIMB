"""One cell of the B1p1 label-efficiency grid for the `no_pretrain_end_to_end` arm.

Fig B1p1 plots four series that are all *frozen probes* re-fit at each label budget
(`scripts/b1_replicates_v2.sh`), plus a fifth, `e2e`, that its legend has carried as
"NOT RUN" -- because that arm needs the ENCODER re-fine-tuned at every budget, which the
frozen sweep structurally cannot produce. That is the only reason it was missing; this
script produces it.

A "cell" is one (label budget n, subsample seed) pair, scored on all 7 tasks. It is the
unit of work because it is also the unit of restart: a box that dies at n=3000 must not
cost us n=100.

Design constraints that are NOT free to change:

  * The encoder is `random_baseline_00/encoder` -- the very weights B1p1's `random`
    series froze. A freshly seeded encoder would make the two series differ in two
    things at once, and the panel's entire claim is that they differ in exactly one.
  * Subsampling goes through `eval_v2._subsample_train` with the same (n, seed), so at
    each grid point this arm trains on the *same molecules* the frozen arms did. A
    private draw would put sampling noise on the gap between the curves.
  * Both `<metric>` and `<metric>_train` must land. B1p1 is a train-vs-test panel; a
    series with test rows and no train rows draws a line in the top half of the figure
    and silently contributes nothing to the mechanism it exists to show. `verify()`
    therefore treats a missing train row as a hard failure, not a warning.

`verified.json` is written only after the summary is checked against the 7 tasks that
were actually requested. File existence proves nothing here: `_write_outputs` rewrites
the summary after EVERY dataset, so a job killed during HIV leaves a perfectly
well-formed CSV describing six tasks.

Usage:
    python scripts/run_b1_e2e_cell.py --n 100 --seed 0 --encoder <dir> \
        --tokenizer <dir> --out_root figure_data/climb_v2_labeleff_v2 \
        [--s3 s3://climb-s3-bucket/experiments/climb_v2_labeleff_v2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import MOLECULENET_TASKS_V2  # noqa: E402
from finetune_e2e_v2 import FT_HPARAMS, evaluate_finetuned  # noqa: E402

# The 7 tasks b1_replicates_v2.sh sweeps, in its order.
B1_TASKS = ["ESOL", "Lipophilicity", "QM7", "BBBP", "BACE", "Tox21", "HIV"]


def verify(cell_dir: Path, tasks) -> dict:
    """Achieved-work check: every task scored, with a train counterpart, non-NaN.

    Deliberately checks the MEAN rows rather than the seed rows -- MEAN is what the
    notebook reads (`query("head_seed=='MEAN'")`), so a cell that satisfies this check
    is exactly a cell the figure can consume.
    """
    report = {"cell": cell_dir.name, "tasks": {}, "ok": True}
    summary = cell_dir / "moleculenet" / "moleculenet_summary.csv"
    if not summary.exists():
        report["ok"] = False
        report["fatal"] = "moleculenet_summary.csv absent"
        return report

    df = pd.read_csv(summary)
    mean = df[df.head_seed.astype(str) == "MEAN"]
    type_map = dict(MOLECULENET_TASKS_V2)
    for t in tasks:
        primary = "roc_auc" if type_map[t] == "classification" else "rmse"
        info = {}
        for want in (primary, f"{primary}_train"):
            hit = mean[(mean.dataset == t) & (mean.main_metric == want)]
            val = float(hit.main_value.iloc[0]) if len(hit) else float("nan")
            info[want] = val
            if not np.isfinite(val):
                report["ok"] = False
        report["tasks"][t] = info
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="label budget; 0 = full train set")
    p.add_argument("--seed", type=int, required=True, help="subsample seed (eval_v2 semantics)")
    p.add_argument("--encoder", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--tasks", nargs="+", default=B1_TASKS)
    p.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="fine-tuning seeds; the end-to-end analogue of eval_v2's --head_seeds, "
                        "which b1_replicates_v2.sh sets to 0 1 2")
    p.add_argument("--epochs", type=int, default=FT_HPARAMS["epochs"])
    p.add_argument("--s3", default=None)
    p.add_argument("--heartbeat", default=None)
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    cell = f"e2e_n{args.n}_s{args.seed}"
    cell_dir = Path(args.out_root) / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    type_map = dict(MOLECULENET_TASKS_V2)
    ds_list = [(t, type_map[t]) for t in args.tasks]
    t_start = time.time()

    def sync():
        if args.s3:
            subprocess.run(["aws", "s3", "sync", str(cell_dir),
                            f"{args.s3.rstrip('/')}/{cell}", "--only-show-errors"], check=False)

    if not args.verify_only:
        # n=0 is the full train set: b1_replicates_v2.sh encodes "no budget" as n=0 and
        # passes no --train_subsample at all, so subsampling must be a genuine no-op here,
        # not a subsample of size 0.
        budget = None if args.n == 0 else args.n
        print(f"===== {cell} :: budget={budget} subsample_seed={args.seed} "
              f"head_seeds={args.head_seeds} =====", flush=True)
        evaluate_finetuned(
            encoder_path=args.encoder, tokenizer_path=args.tokenizer,
            output_dir=str(cell_dir / "moleculenet"), seeds=list(args.head_seeds),
            datasets=ds_list, cv_folds=None, train_subsample=budget,
            subsample_seed=args.seed, epochs=args.epochs,
            heartbeat_path=args.heartbeat or str(cell_dir / "heartbeat.json"),
        )
        sync()

    rep = verify(cell_dir, args.tasks)
    (cell_dir / "verification_report.json").write_text(json.dumps(rep, indent=2))

    (cell_dir / "metadata.json").write_text(json.dumps({
        "cell": cell,
        "figure": "B1p1",
        "series": "e2e",
        "run_type": "no_pretrain_end_to_end",
        "regime": "no_pretrain_e2e",
        "description": ("random-init ~41M ModernBERT fine-tuned END-TO-END at label budget "
                        f"n={args.n or 'full'}; the unfrozen twin of the `random` series, "
                        "which freezes these same weights and fits an MLP probe"),
        "encoder_source": args.encoder,
        "train_subsample": args.n or None,
        "subsample_seed": args.seed,
        "head_seeds": list(args.head_seeds),
        "tasks": list(args.tasks),
        "hyperparameters": {
            "lr": FT_HPARAMS["lr"], "weight_decay": FT_HPARAMS["weight_decay"],
            "max_epochs": args.epochs, "early_stopping_patience": FT_HPARAMS["patience"],
            "batch_size": FT_HPARAMS["batch_size"], "max_length": FT_HPARAMS["max_length"],
            "optimizer": "AdamW", "precision": "bf16 autocast (fp32 master weights)",
            "pooling": "masked mean", "head": "Linear(hidden, n_outputs)",
        },
        "deviations_from_frozen_protocol": [
            "encoder is fine-tuned, not frozen (the point of the arm)",
            "head is Linear, not the frozen arm's MLP -- an MLP on top of a trainable "
            "encoder adds capacity the frozen arm cannot express, which would confound "
            "frozen-vs-fine-tuned with head capacity",
            "no feature standardization: the frozen arm z-scores fixed embeddings, which "
            "is undefined for representations that move every step",
            "MEAN/STD are over fine-tuning seeds, where the frozen arm's are over head seeds",
        ],
        "elapsed_seconds": round(time.time() - t_start, 1),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    if rep["ok"]:
        (cell_dir / "verified.json").write_text(json.dumps({
            "cell": cell, "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "criterion": "every requested task has a finite MEAN row for BOTH <metric> and "
                         "<metric>_train (the two rows Fig B1p1 reads per point)",
            "tasks": list(args.tasks),
        }, indent=2))
        print(f"[b1_e2e] VERIFIED {cell}", flush=True)
    else:
        (cell_dir / "verified.json").unlink(missing_ok=True)
        print(f"[b1_e2e] NOT VERIFIED {cell}: {json.dumps(rep)}", flush=True)

    sync()
    sys.exit(0 if rep["ok"] else 3)


if __name__ == "__main__":
    main()

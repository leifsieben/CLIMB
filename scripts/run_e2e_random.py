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

from config import MOLECULENET_TASKS
from finetune import FT_HPARAMS, evaluate_finetuned

CORE_TASKS = ["ESOL", "BACE", "Tox21", "QM7", "HIV"]
CV_FOLDS = 5
SUBSAMPLE_SEED = 0


def _expected_rows(task: str, scheme: str) -> int:

    from evaluate import _load_moleculenet, _load_moleculenet_full
    if scheme == "moleculenet_cv":
        _, y = _load_moleculenet_full(task)
    else:
        _, _, _, _, _, y = _load_moleculenet(task)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    return int(np.isfinite(y).sum())


def verify(run_dir: Path, tasks=CORE_TASKS) -> dict:
    """Achieved-work check. Returns a report; `ok` is the only thing callers may trust."""
    report = {"run_id": run_dir.name, "schemes": {}, "ok": True}
    for scheme in ("moleculenet", "moleculenet_cv"):
        d = run_dir / scheme
        info = {"missing_metric": [], "row_counts": {}, "row_count_mismatch": []}
        suite_path = d / "suite_summary.json"
        preds_path = d / "test_predictions.csv"
        if not suite_path.exists() or not preds_path.exists():
            info["fatal"] = "suite_summary.json or test_predictions.csv absent"
            report["schemes"][scheme] = info
            report["ok"] = False
            continue
        suite = json.loads(suite_path.read_text())
        for t in tasks:
            key = "HIV_nef1_MEAN" if t == "HIV" else f"{t}_MEAN"
            v = suite.get(key)
            if v is None or not np.isfinite(v):
                info["missing_metric"].append(key)
        preds = pd.read_csv(preds_path)
        counts = preds.groupby("dataset").size().to_dict()
        for t in tasks:
            got = int(counts.get(t, 0))
            want = _expected_rows(t, scheme)
            info["row_counts"][t] = {"got": got, "want": want}
            if got != want:
                info["row_count_mismatch"].append(t)
        if info["missing_metric"] or info["row_count_mismatch"]:
            report["ok"] = False
        report["schemes"][scheme] = info
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replicate", type=int, required=True, help="0/1/2 -> e2e_random_0N")
    p.add_argument("--encoder", required=True, help="random_baseline_0N/encoder")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--datasets", nargs="+", default=CORE_TASKS)
    p.add_argument("--holdout_seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="matches evaluate's head_seeds=[0,1,2]; hold-out STD is over these")
    p.add_argument("--cv_seeds", type=int, nargs="+", default=[0],
                   help="fine-tune seeds averaged into each fold's prediction; the frozen "
                        "arm uses 3 — 1 here is the documented compute scope-down")
    p.add_argument("--epochs", type=int, default=FT_HPARAMS["epochs"])
    p.add_argument("--schemes", nargs="+", default=["moleculenet", "moleculenet_cv"])
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    run_id = f"e2e_random_{args.replicate:02d}"
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    type_map = dict(MOLECULENET_TASKS)
    ds_list = [(n, type_map[n]) for n in args.datasets]
    t_start = time.time()


    if not args.verify_only:
        (run_dir / "run_status.json").write_text(json.dumps(
            {"status": "running", "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        for scheme in args.schemes:
            cv = CV_FOLDS if scheme == "moleculenet_cv" else None
            seeds = args.cv_seeds if cv else args.holdout_seeds
            print(f"\n===== {run_id} :: {scheme} (cv_folds={cv}, seeds={seeds}) =====", flush=True)
            evaluate_finetuned(
                encoder_path=args.encoder, tokenizer_path=args.tokenizer,
                output_dir=str(run_dir / scheme), seeds=list(seeds), datasets=ds_list,
                cv_folds=cv, subsample_seed=SUBSAMPLE_SEED, epochs=args.epochs,
                heartbeat_path=str(run_dir / "heartbeat.json"),
            )


    rep = verify(run_dir, tasks=args.datasets)
    (run_dir / "verification_report.json").write_text(json.dumps(rep, indent=2))

    (run_dir / "config.yaml").write_text(
        "\n".join([
            f"run_id: {run_id}",
            f"tokenizer_path: {json.dumps(args.tokenizer)}",
            "model:", "  hidden_size: 512", "  num_hidden_layers: 12",
            "  num_attention_heads: 8", "  intermediate_size: 1536",
            "  max_position_embeddings: 256", "  vocab_size: 1000",
            "selection:", f"  pretraining_seed: {args.replicate}",
            "  run_type: no_pretrain_end_to_end",
            "evaluation:", "  freeze_encoder: false", "  pool: mean",
            "  standardize: none", "  head: linear_e2e",
            f"  max_length: {FT_HPARAMS['max_length']}",
            "  head_seeds:", *[f"  - {s}" for s in args.holdout_seeds],
            "",
        ]))

    (run_dir / "metadata.json").write_text(json.dumps({
        "run_id": run_id,
        "run_type": "no_pretrain_end_to_end",
        "regime": "no_pretrain_e2e",
        "description": ("random-init ~41M ModernBERT encoder fine-tuned END-TO-END on each "
                        "downstream task; the unfrozen twin of random_baseline_"
                        f"{args.replicate:02d} (identical starting weights)"),
        "encoder_source": args.encoder,
        "encoder_seed": args.replicate,
        "pretraining_forward_passes": 0,
        "code": {"module": "finetune", "runner": "scripts/run_e2e_random.py"},
        "hyperparameters": {
            "lr": FT_HPARAMS["lr"], "weight_decay": FT_HPARAMS["weight_decay"],
            "max_epochs": args.epochs, "early_stopping_patience": FT_HPARAMS["patience"],
            "batch_size": FT_HPARAMS["batch_size"], "max_length": FT_HPARAMS["max_length"],
            "optimizer": "AdamW", "precision": "bf16 autocast (fp32 master weights)",
            "pooling": "masked mean", "head": "Linear(hidden, n_outputs)",
            "loss": "per-column masked BCEWithLogits (clf) / MSE (reg)",
            "target_standardization": ("none — DeepChem molnet already applies a train-fitted "
                                       "NormalizationTransformer to the regression sets"),
        },
        "protocol": {
            "holdout_seeds": list(args.holdout_seeds),
            "cv_seeds": list(args.cv_seeds),
            "cv_folds": CV_FOLDS,
            "subsample_seed": SUBSAMPLE_SEED,
            "tasks": list(args.datasets),
            "prediction_values": "raw logits (classification) / values (regression)",
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    (run_dir / "run_status.json").write_text(json.dumps({
        "status": "ok" if rep["ok"] else "incomplete",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }, indent=2))

    if rep["ok"]:


        (run_dir / "verified.json").write_text(json.dumps({
            "run_id": run_id, "budget_fp": 0, "final_fp": 0, "fraction": 1.0,
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backfilled": False, "route": "anchor",
            "criterion": "all core tasks scored in both schemes; one prediction row per "
                         "non-NaN label entry in each scheme",
        }, indent=2))
        print(f"[e2e] VERIFIED {run_id}", flush=True)
    else:
        (run_dir / "verified.json").unlink(missing_ok=True)
        print(f"[e2e] NOT VERIFIED {run_id}: {json.dumps(rep)}", flush=True)

    sys.exit(0 if rep["ok"] else 3)


if __name__ == "__main__":
    main()

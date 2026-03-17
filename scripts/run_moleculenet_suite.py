#!/usr/bin/env python3
"""Run the full MoleculeNet evaluation suite for a pretrained encoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_manifest import DEFAULT_MOLECULENET_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MoleculeNet evaluation suite")
    parser.add_argument("--pretrained_model", required=True, help="Path to pretrained model/encoder")
    parser.add_argument("--output", required=True, help="Output directory for the suite")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder during downstream fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_MOLECULENET_DATASETS)
    parser.add_argument("--skip_existing", action="store_true", help="Skip datasets with existing results.json")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []

    for dataset_name in args.datasets:
        dataset_dir = output_root / dataset_name
        result_file = dataset_dir / "results.json"
        if args.skip_existing and result_file.exists():
            completed.append(dataset_name)
            continue

        cmd = [
            "python3",
            "evaluate_model.py",
            "--pretrained_model",
            args.pretrained_model,
            "--dataset",
            "moleculenet",
            "--dataset_name",
            dataset_name,
            "--output",
            str(dataset_dir),
            "--tokenizer",
            args.tokenizer,
            "--num_epochs",
            str(args.num_epochs),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.learning_rate),
        ]
        if args.freeze_encoder:
            cmd.append("--freeze_encoder")
        subprocess.run(cmd, check=True)
        completed.append(dataset_name)

    subprocess.run(
        [
            "python3",
            "scripts/aggregate_moleculenet.py",
            "--root",
            str(output_root),
            "--output",
            str(output_root / "moleculenet_summary.csv"),
        ],
        check=True,
    )

    payload = {
        "pretrained_model": args.pretrained_model,
        "tokenizer": args.tokenizer,
        "datasets": completed,
        "summary_csv": str(output_root / "moleculenet_summary.csv"),
    }
    with (output_root / "suite_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Completed MoleculeNet suite for {len(completed)} dataset(s)")


if __name__ == "__main__":
    main()

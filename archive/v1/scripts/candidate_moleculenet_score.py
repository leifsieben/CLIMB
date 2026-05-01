#!/usr/bin/env python3
"""Compute a provisional aggregate score from raw MoleculeNet metrics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


HIGHER_IS_BETTER = {"roc_auc"}
LOWER_IS_BETTER = {"rmse"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a candidate aggregate MoleculeNet score")
    parser.add_argument("--input", required=True, help="CSV from aggregate_experiment_results.py")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    by_dataset = defaultdict(list)
    for row in rows:
        if not row.get("dataset") or not row.get("main_value"):
            continue
        by_dataset[row["dataset"]].append(float(row["main_value"]))

    dataset_stats = {}
    for dataset, values in by_dataset.items():
        sigma = pstdev(values) if len(values) > 1 else 0.0
        dataset_stats[dataset] = (mean(values), sigma)

    per_run = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset")
        metric = row.get("main_metric")
        if not dataset or not row.get("main_value") or dataset not in dataset_stats:
            continue
        mu, sigma = dataset_stats[dataset]
        value = float(row["main_value"])
        z = 0.0 if sigma == 0 else (value - mu) / sigma
        if metric in LOWER_IS_BETTER:
            z *= -1.0
        per_run[row["run_id"]].append(z)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "candidate_score", "num_datasets"])
        writer.writeheader()
        for run_id, scores in sorted(per_run.items()):
            writer.writerow(
                {
                    "run_id": run_id,
                    "candidate_score": round(mean(scores), 6) if scores else "",
                    "num_datasets": len(scores),
                }
            )

    print(f"Wrote candidate scores to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate MoleculeNet evaluation results into a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing evaluation outputs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    result_files = list(root.rglob("results.json"))
    rows = []
    for path in result_files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        task_type = data.get("task_type")
        metrics = data.get("metrics", {})
        main_metric = "roc_auc" if task_type == "classification" else "rmse"
        rows.append(
            {
                "dataset": data.get("dataset"),
                "task_type": task_type,
                "main_metric": main_metric,
                "main_value": metrics.get(main_metric),
                "pretrained_model": data.get("pretrained_model"),
                "results_path": str(path),
            }
        )

    rows.sort(key=lambda r: (r.get("pretrained_model", ""), r.get("dataset", "")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "task_type", "main_metric", "main_value", "pretrained_model", "results_path"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

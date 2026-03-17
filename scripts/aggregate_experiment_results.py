#!/usr/bin/env python3
"""Aggregate manifest runs, training metadata, and MoleculeNet outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CLIMB experiment artifacts")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    manifest = _load_json(Path(args.manifest))
    rows: List[Dict] = []
    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "metrics.jsonl"
        suite_path = Path(run["evaluation_output_dir"]) / "moleculenet_summary.csv"

        meta = _load_json(metadata_path) if metadata_path.exists() else {}
        suite_rows = []
        if suite_path.exists():
            with suite_path.open() as f:
                suite_rows = list(csv.DictReader(f))

        if not suite_rows:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "run_type": run["run_type"],
                    "stage": run["stage"],
                    "dataset": None,
                    "main_metric": None,
                    "main_value": None,
                    "output_dir": run["output_dir"],
                    "backup_s3_uri": run.get("backup_s3_uri"),
                    "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                    "selection": json.dumps(run.get("selection", {}), sort_keys=True),
                    "token_budget_total": meta.get("metadata", {}).get("token_budget_total"),
                }
            )
            continue

        for suite_row in suite_rows:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "run_type": run["run_type"],
                    "stage": run["stage"],
                    "dataset": suite_row.get("dataset"),
                    "main_metric": suite_row.get("main_metric"),
                    "main_value": suite_row.get("main_value"),
                    "output_dir": run["output_dir"],
                    "backup_s3_uri": run.get("backup_s3_uri"),
                    "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                    "selection": json.dumps(run.get("selection", {}), sort_keys=True),
                    "token_budget_total": meta.get("metadata", {}).get("token_budget_total"),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_type",
        "stage",
        "dataset",
        "main_metric",
        "main_value",
        "output_dir",
        "backup_s3_uri",
        "metrics_path",
        "metadata_path",
        "selection",
        "token_budget_total",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

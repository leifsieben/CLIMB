#!/usr/bin/env python3
"""Aggregate metrics.jsonl files into a single CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing run outputs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    metrics_files = list(root.rglob("metrics.jsonl"))

    rows = []
    for path in metrics_files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["metrics_path"] = str(path)
                rows.append(rec)

    rows.sort(key=lambda r: (r.get("run_id", ""), r.get("phase", ""), r.get("tokens_seen", 0)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "phase",
        "global_step",
        "tokens_seen",
        "loss",
        "learning_rate",
        "epoch",
        "timestamp",
        "metrics_path",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Populate runtime estimates in a manifest from completed smoke runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _load_metrics(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _estimate_tokens_per_second(run_dir: Path) -> Optional[float]:
    metrics = _load_metrics(run_dir / "metrics.jsonl")
    if len(metrics) < 2:
        return None
    first = metrics[0]
    last = metrics[-1]
    dt = last["timestamp"] - first["timestamp"]
    if dt <= 0:
        return None
    return (last.get("tokens_seen", 0) - first.get("tokens_seen", 0)) / dt


def _estimate_eval_hours(eval_dir: Path) -> Optional[float]:
    suite = eval_dir / "suite_summary.json"
    if not suite.exists():
        return None
    mtime = suite.stat().st_mtime
    results = list(eval_dir.glob("*/results.json"))
    if not results:
        return None
    first = min(result.stat().st_mtime for result in results)
    return max(0.0, mtime - first) / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate manifest runtime estimates from smoke runs")
    parser.add_argument("--manifest", required=True, help="Manifest JSON to update in place")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)

    smoke_runs = [run for run in manifest["runs"] if run["stage"] == "smoke"]
    tps_values = []
    eval_values = []
    for run in smoke_runs:
        run_dir = Path(run["output_dir"])
        tps = _estimate_tokens_per_second(run_dir)
        eval_hours = _estimate_eval_hours(Path(run["evaluation_output_dir"]))
        if tps:
            tps_values.append(tps)
        if eval_hours is not None:
            eval_values.append(eval_hours)

    calibration = manifest.setdefault("calibration", {})
    if tps_values:
        calibration["tokens_per_second"] = sum(tps_values) / len(tps_values)
    if eval_values:
        calibration["evaluation_hours"] = sum(eval_values) / len(eval_values)

    tokens_per_second = calibration.get("tokens_per_second")
    evaluation_hours = calibration.get("evaluation_hours")
    for run in manifest["runs"]:
        token_budget = run["pretrain_config"]["compute_budget"].get("total_tokens")
        if token_budget and tokens_per_second:
            pretrain_hours = token_budget / tokens_per_second / 3600.0
            run["runtime_estimate"] = {
                "pretrain_hours": round(pretrain_hours, 3),
                "evaluation_hours": evaluation_hours,
                "bundle_hours": round(pretrain_hours + (evaluation_hours or 0.0), 3),
                "estimate_basis": "smoke_calibrated",
            }

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Monitor training progress from metrics.jsonl and backup status."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import find_latest_checkpoint


def load_last_records(path: Path, n: int = 20) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-n:]


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days > 0:
        return f"{days}d {hrs}h"
    if hrs > 0:
        return f"{hrs}h {mins}m"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Run directory with metrics.jsonl")
    parser.add_argument("--metrics", default="metrics.jsonl", help="Metrics JSONL filename")
    parser.add_argument("--metadata", default="metadata.json", help="Metadata JSON filename")
    parser.add_argument("--token_budget", type=int, help="Override token budget")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / args.metrics
    metadata_path = run_dir / args.metadata
    backup_path = run_dir / "backup_status.json"

    records = load_last_records(metrics_path, n=20)
    if not records:
        print("No metrics found.")
        return

    last = records[-1]
    tokens_seen = last.get("tokens_seen", 0)
    loss = last.get("loss")
    phase = last.get("phase")
    run_id = last.get("run_id")

    token_budget = args.token_budget
    if token_budget is None and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            token_budget = meta.get("metadata", {}).get("token_budget_total") or meta.get("token_budget_total")
        except Exception:
            token_budget = None

    eta = "unknown"
    if len(records) >= 2 and token_budget:
        first = records[0]
        dt = last["timestamp"] - first["timestamp"]
        dt = max(dt, 1e-6)
        tokens_rate = (last.get("tokens_seen", 0) - first.get("tokens_seen", 0)) / dt
        if tokens_rate > 0:
            remaining = max(0, token_budget - tokens_seen)
            eta = format_eta(remaining / tokens_rate)

    backup_status = "unknown"
    if backup_path.exists():
        try:
            backup = json.loads(backup_path.read_text())
            backup_status = f"{backup.get('last_sync_utc')} -> {backup.get('s3_path')}"
        except Exception:
            backup_status = "unreadable"

    current_checkpoint = find_latest_checkpoint(str(run_dir)) or "none"
    eval_dir = run_dir / "moleculenet"
    eval_count = len(list(eval_dir.glob("*/results.json"))) if eval_dir.exists() else 0
    suite_done = (eval_dir / "suite_summary.json").exists()

    print(f"run_id={run_id} phase={phase}")
    print(f"tokens_seen={tokens_seen} loss={loss}")
    if token_budget:
        pct = 100.0 * tokens_seen / token_budget
        print(f"token_budget={token_budget} ({pct:.2f}%) eta={eta}")
    else:
        print("token_budget=unknown")
    print(f"current_checkpoint={current_checkpoint}")
    print(f"evaluation_progress={eval_count}")
    print(f"evaluation_complete={suite_done}")
    print(f"backup_status={backup_status}")


if __name__ == "__main__":
    main()

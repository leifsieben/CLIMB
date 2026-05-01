#!/usr/bin/env python3
"""Periodic S3 sync for run directories."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def write_status(run_dir: Path, s3_dest: str, status: str) -> None:
    payload = {
        "last_sync_utc": datetime.now(timezone.utc).isoformat(),
        "s3_path": s3_dest,
        "status": status,
    }
    with (run_dir / "backup_status.json").open("w") as f:
        json.dump(payload, f, indent=2)


def run_sync(run_dir: Path, s3_dest: str) -> None:
    cmd = ["aws", "s3", "sync", str(run_dir), s3_dest, "--no-progress"]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a run directory to S3")
    parser.add_argument("--run_dir", required=True, help="Local run directory")
    parser.add_argument("--s3_dest", required=True, help="Destination S3 prefix")
    parser.add_argument("--interval_seconds", type=int, default=0, help="Repeat sync on this interval; 0 means once")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.interval_seconds <= 0:
        write_status(run_dir, args.s3_dest, "running")
        run_sync(run_dir, args.s3_dest)
        write_status(run_dir, args.s3_dest, "ok")
        return

    while True:
        try:
            write_status(run_dir, args.s3_dest, "running")
            run_sync(run_dir, args.s3_dest)
            write_status(run_dir, args.s3_dest, "ok")
        except Exception as exc:
            write_status(run_dir, args.s3_dest, f"error: {exc}")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

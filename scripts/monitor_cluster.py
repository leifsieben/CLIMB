#!/usr/bin/env python3
"""Monitor manifest runs across SSH-accessible workers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _manifest_runs(manifest: Dict) -> Dict[str, Dict]:
    return {run["run_id"]: run for run in manifest["runs"]}


def _remote_monitor(worker: Dict, remote_run_dir: str) -> Dict[str, str]:
    workspace_root = worker["workspace_root"]
    python_bin = worker.get("python_bin", "python3")
    user = worker.get("user", "ec2-user")
    key_path = worker.get("key_path")

    remote_script = (
        f"cd {workspace_root} && "
        f"{python_bin} scripts/monitor_run.py --run_dir {remote_run_dir} ; "
        f"echo __GPU__ ; "
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null || true ; "
        "echo __DISK__ ; "
        f"df -h {remote_run_dir} 2>/dev/null | tail -n 1 || true"
    )
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", f"{user}@{worker['host']}", remote_script]
    if key_path:
        cmd[1:1] = ["-i", key_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload = {"stdout": result.stdout, "stderr": result.stderr, "returncode": str(result.returncode)}
    return payload


def _parse_output(output: Dict[str, str]) -> Dict[str, str]:
    if output["returncode"] != "0":
        return {"status": "ssh_error", "details": output["stderr"].strip() or output["stdout"].strip()}

    lines = [line.strip() for line in output["stdout"].splitlines() if line.strip()]
    parsed: Dict[str, str] = {"status": "ok"}
    mode = "monitor"
    gpu_lines: List[str] = []
    disk_lines: List[str] = []
    for line in lines:
        if line == "__GPU__":
            mode = "gpu"
            continue
        if line == "__DISK__":
            mode = "disk"
            continue
        if mode == "monitor" and "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
        elif mode == "gpu":
            gpu_lines.append(line)
        elif mode == "disk":
            disk_lines.append(line)
    parsed["gpu"] = " | ".join(gpu_lines)
    parsed["disk"] = " | ".join(disk_lines)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor manifest runs across workers")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--cluster_config", required=True, help="Cluster config JSON")
    parser.add_argument("--output_json", help="Optional JSON snapshot path")
    parser.add_argument("--output_csv", help="Optional CSV snapshot path")
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    cluster = _load_json(args.cluster_config)
    runs_by_id = _manifest_runs(manifest)

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workers": [],
    }
    rows: List[Dict[str, str]] = []

    for worker in cluster.get("workers", []):
        worker_payload = {"name": worker["name"], "host": worker["host"], "runs": []}
        for run_id in worker.get("run_ids", []):
            run = runs_by_id[run_id]
            remote_dir = str(Path(worker["run_root"]) / Path(run["output_dir"]).name)
            parsed = _parse_output(_remote_monitor(worker, remote_dir))
            record = {
                "worker_name": worker["name"],
                "host": worker["host"],
                "run_id": run_id,
                "remote_run_dir": remote_dir,
                **parsed,
            }
            worker_payload["runs"].append(record)
            rows.append(record)
        snapshot["workers"].append(worker_payload)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(snapshot, f, indent=2)

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    for row in rows:
        print(
            f"{row['worker_name']} {row['run_id']} status={row.get('status')} "
            f"phase={row.get('phase')} tokens={row.get('tokens_seen')} eta={row.get('eta')} "
            f"backup={row.get('backup_status')}"
        )


if __name__ == "__main__":
    main()

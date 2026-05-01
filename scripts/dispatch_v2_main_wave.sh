#!/usr/bin/env bash
# Dispatch the v2 main wave: SSH each worker, build a run_id list per cluster_config_v2.json,
# and start launch_v2_wave.py in a nohup'd background process.
#
# Usage:
#   bash scripts/dispatch_v2_main_wave.sh
#
# The cluster config is at configs/cluster_config_v2.json.

set -euo pipefail

CLUSTER_CONFIG="${1:-configs/cluster_config_v2.json}"
KEY_PATH="$(pwd)/climb-gpu-key.pem"

if [ ! -f "$CLUSTER_CONFIG" ]; then
  echo "Cluster config not found: $CLUSTER_CONFIG"
  exit 1
fi

python3 - <<EOF
import json
import os
import subprocess
import sys
from pathlib import Path

cfg = json.loads(Path("$CLUSTER_CONFIG").read_text())
key = "$KEY_PATH"

for w in cfg["workers"]:
    name = w["name"]
    host = w["host"]
    user = w["user"]
    workspace = w["workspace_root"]
    python_bin = w["python_bin"]
    run_ids = w["run_ids"]
    if not run_ids:
        print(f"[{name}] no runs; skipping")
        continue

    run_args = " ".join(f"--run_id {rid}" for rid in run_ids)
    remote_cmd = (
        f"set -e; "
        f"mkdir -p /home/ec2-user/artifacts/robust_matrix_v2_logs; "
        f"cd {workspace}; "
        f"nohup {python_bin} scripts/launch_v2_wave.py "
        f"--manifest experiments/robust_matrix_v2/manifest.json "
        f"{run_args} "
        f"--worker_name {name} "
        f">> /home/ec2-user/artifacts/robust_matrix_v2_logs/main_wave.log 2>&1 < /dev/null & "
        f"echo \"pid \\$!\""
    )

    print(f"[{name}] dispatching {len(run_ids)} runs ...")
    res = subprocess.run([
        "ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        f"{user}@{host}", remote_cmd,
    ], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        print(f"[{name}] FAILED: {res.stderr.strip()}")
    else:
        print(f"[{name}] {res.stdout.strip()}")
EOF

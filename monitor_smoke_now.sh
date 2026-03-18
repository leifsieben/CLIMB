#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-60}"
MANIFEST="${MANIFEST:-/tmp/climb_manifest_20260318.json}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-./configs/cluster_config_smoke_20260317.json}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Generate it with:" >&2
  echo "  .venv_sanity/bin/python scripts/generate_experiment_manifest.py --spec configs/experiment_spec_example.yaml --output /tmp/climb_manifest_20260318.json" >&2
  exit 1
fi

if [[ ! -f "$CLUSTER_CONFIG" ]]; then
  echo "Cluster config not found: $CLUSTER_CONFIG" >&2
  exit 1
fi

trap 'echo; echo "Stopped."; exit 0' INT TERM

while true; do
  clear
  echo "============================================================"
  echo "CLIMB Smoke Monitor $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"

  python3 - "$MANIFEST" "$CLUSTER_CONFIG" <<'PY' | while IFS=$'\t' read -r worker host user key_path workspace_root run_root run_id output_dir eval_output_dir; do
import json
import os
import sys

manifest = json.load(open(sys.argv[1]))
cluster = json.load(open(sys.argv[2]))
runs = {run["run_id"]: run for run in manifest["runs"]}

for worker in cluster["workers"]:
    for run_id in worker.get("run_ids", []):
        run = runs[run_id]
        print(
            "\t".join(
                [
                    worker["name"],
                    worker["host"],
                    worker.get("user", "ec2-user"),
                    worker.get("key_path", ""),
                    worker.get("workspace_root", ""),
                    worker["run_root"],
                    run_id,
                    run["output_dir"],
                    run.get("evaluation_output_dir", ""),
                ]
            )
        )
PY
    if [[ "$output_dir" = /* ]]; then
      RUN_DIR="$output_dir"
    else
      RUN_DIR="${workspace_root}/${output_dir}"
    fi
    if [[ -n "$eval_output_dir" ]]; then
      if [[ "$eval_output_dir" = /* ]]; then
        EVAL_DIR="$eval_output_dir"
      else
        EVAL_DIR="${workspace_root}/${eval_output_dir}"
      fi
    else
      EVAL_DIR="${RUN_DIR}/moleculenet"
    fi
    LOG_FILE="${run_root}_logs/${run_id}.log"

    echo
    echo "[$worker] $run_id @ $host"
    ssh -i "$key_path" -o StrictHostKeyChecking=no "${user}@${host}" \
      RUN_DIR="$RUN_DIR" LOG_FILE="$LOG_FILE" 'bash -s' <<'EOF'
set -euo pipefail
STATUS=unknown
if [ -f "$EVAL_DIR/suite_summary.json" ] || [ -f "$RUN_DIR/evaluations/suite_summary.json" ]; then
  STATUS=success
elif pgrep -af 'launch_experiment_wave|pretrain_pipeline|run_moleculenet_suite|evaluate_model' >/dev/null 2>&1; then
  STATUS=running
elif [ -f "$RUN_DIR/run_context.json" ] || [ -f "$LOG_FILE" ]; then
  STATUS=stopped_or_failed
else
  STATUS=not_started
fi
echo "status=$STATUS"
echo "run_dir=$RUN_DIR"
if [ -f "$RUN_DIR/backup_status.json" ]; then
  BACKUP_STATUS="$(tr '\n' ' ' < "$RUN_DIR/backup_status.json" | cut -c1-180)"
  echo "backup_status=$BACKUP_STATUS"
fi
if [ -f "$RUN_DIR/metadata.json" ]; then
  echo "metadata_present=yes"
fi
if [ -f "$RUN_DIR/training_results.json" ]; then
  echo "training_results_present=yes"
fi
echo "processes:"
pgrep -af 'launch_experiment_wave|pretrain_pipeline|run_moleculenet_suite|evaluate_model' || true
echo "last_log_lines:"
tail -n 8 "$LOG_FILE" 2>/dev/null || echo '(no log yet)'
EOF
  done

  echo
  echo "Next poll in ${INTERVAL}s"
  sleep "$INTERVAL"
done

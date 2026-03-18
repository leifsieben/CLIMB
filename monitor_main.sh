#!/usr/bin/env bash
# Monitor the main experiment wave across all 3 workers.
# Usage:  bash monitor_main.sh [poll_interval_seconds]
set -euo pipefail

INTERVAL="${1:-60}"
MANIFEST="${MANIFEST:-/home/ec2-user/artifacts/robust_matrix/manifest.json}"
KEY="/Users/lsieben/VSCode/CLIMB/climb-gpu-key.pem"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-/Users/lsieben/VSCode/CLIMB/configs/cluster_config_main_20260318.json}"
LOG_PREFIX="main_wave"

declare -A WORKER_HOSTS=(
  [worker0]="3.89.64.232"
  [worker1]="54.89.239.106"
  [worker2]="18.206.240.192"
)

trap 'echo; echo "Stopped."; exit 0' INT TERM

while true; do
  clear
  echo "============================================================"
  echo "CLIMB Main Experiment Monitor  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"

  # Pull run assignments from local cluster config
  python3 - "$CLUSTER_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for w in cfg["workers"]:
    run_ids = w.get("run_ids", [])
    print(f"  {w['name']} ({w['host']}): {len(run_ids)} assigned runs")
    for rid in run_ids[:3]:
        print(f"    {rid}")
    if len(run_ids) > 3:
        print(f"    ... +{len(run_ids)-3} more")
PY

  echo ""
  echo "------------------------------------------------------------"

  for worker in worker0 worker1 worker2; do
    host="${WORKER_HOSTS[$worker]}"
    LOG_FILE="/home/ec2-user/artifacts/robust_matrix_logs/${LOG_PREFIX}_${worker}.log"
    RUN_ROOT="/home/ec2-user/artifacts/robust_matrix"

    echo ""
    echo "[$worker] @ $host"

    # One SSH call: process status + GPU + last meaningful log lines
    ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
        "ec2-user@${host}" \
        LOG_FILE="$LOG_FILE" RUN_ROOT="$RUN_ROOT" 'bash -s' << 'REMOTE' 2>/dev/null || echo "  (ssh error)"

# Launcher process
LAUNCHER_PID=$(pgrep -f "launch_experiment_wave" 2>/dev/null || true)
PRETRAIN_PID=$(pgrep -f "pretrain_pipeline" 2>/dev/null || true)
EVAL_PID=$(pgrep -f "run_moleculenet_suite|evaluate_model" 2>/dev/null || true)

if [[ -n "$PRETRAIN_PID" ]]; then
  PROC_STATUS="pretraining (pid $PRETRAIN_PID)"
elif [[ -n "$EVAL_PID" ]]; then
  PROC_STATUS="evaluating  (pid $EVAL_PID)"
elif [[ -n "$LAUNCHER_PID" ]]; then
  PROC_STATUS="launcher idle (pid $LAUNCHER_PID)"
else
  PROC_STATUS="no active process"
fi
echo "  status : $PROC_STATUS"

# Count completed runs (have metadata.json or suite_summary.json)
DONE=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | while read d; do
  [[ -f "$d/metadata.json" ]] && echo ok || [[ -f "$d/moleculenet/suite_summary.json" ]] && echo ok; true
done | wc -l)
TOTAL=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "  runs   : $DONE/$TOTAL complete in $RUN_ROOT"

# GPU
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
echo "  GPU    : ${GPU} (util%, mem_used MB, mem_total MB)"

# Disk
DISK=$(df -h / 2>/dev/null | tail -1 | awk '{print $3"/"$2" used ("$5")"}')
echo "  disk   : $DISK"

# Last non-upload meaningful log lines
if [[ -f "$LOG_FILE" ]]; then
  echo "  recent log:"
  grep -av "upload:\|backup_status" "$LOG_FILE" 2>/dev/null | tail -6 | sed 's/^/    /'
fi
REMOTE

  done

  echo ""
  echo "------------------------------------------------------------"
  echo "Backup check (S3 last sync times):"
  aws s3 ls s3://climb-s3-bucket/experiments/robust_matrix/ 2>/dev/null | awk '{print "  "$NF}' | head -10
  echo ""
  echo "Next poll in ${INTERVAL}s  (Ctrl-C to stop)"
  sleep "$INTERVAL"
done

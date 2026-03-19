#!/usr/bin/env bash
# Monitor the main experiment wave across all 3 workers.
# Usage:  bash monitor_main.sh [poll_interval_seconds]

INTERVAL="${1:-60}"
KEY="/Users/lsieben/VSCode/CLIMB/climb-gpu-key.pem"
LOG_PREFIX="main_wave"

WORKERS="worker0 worker1 worker2"
HOSTS_worker0="3.89.64.232"
HOSTS_worker1="54.89.239.106"
HOSTS_worker2="18.206.240.192"

trap 'echo; echo "Stopped."; exit 0' INT TERM

_get_host() {
  case "$1" in
    worker0) echo "$HOSTS_worker0" ;;
    worker1) echo "$HOSTS_worker1" ;;
    worker2) echo "$HOSTS_worker2" ;;
  esac
}

while true; do
  clear
  echo "============================================================"
  echo "CLIMB Main Experiment Monitor  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"

  for worker in $WORKERS; do
    host=$(_get_host "$worker")
    LOG_FILE="/home/ec2-user/artifacts/robust_matrix_logs/${LOG_PREFIX}_${worker}.log"
    RUN_ROOT="/home/ec2-user/artifacts/robust_matrix"

    echo ""
    echo "[$worker] @ $host"

    ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
        "ec2-user@${host}" \
        "LOG_FILE=$LOG_FILE RUN_ROOT=$RUN_ROOT bash -s" << 'REMOTE' 2>/dev/null || echo "  (ssh error)"

PRETRAIN_PID=$(pgrep -f "pretrain_pipeline" 2>/dev/null | head -1 || true)
LAUNCHER_PID=$(pgrep -f "launch_experiment_wave" 2>/dev/null | head -1 || true)
EVAL_PID=$(pgrep -f "run_moleculenet_suite|evaluate_model" 2>/dev/null | head -1 || true)

if [ -n "$PRETRAIN_PID" ]; then
  PROC_STATUS="pretraining (pid $PRETRAIN_PID)"
elif [ -n "$EVAL_PID" ]; then
  PROC_STATUS="evaluating  (pid $EVAL_PID)"
elif [ -n "$LAUNCHER_PID" ]; then
  PROC_STATUS="launcher idle (pid $LAUNCHER_PID)"
else
  PROC_STATUS="no active process"
fi
echo "  status : $PROC_STATUS"

# Count dirs that have metadata.json (pretrain done) or suite_summary.json (eval done)
PRETRAIN_DONE=$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name "metadata.json" 2>/dev/null | wc -l | tr -d ' ')
EVAL_DONE=$(find "$RUN_ROOT" -mindepth 3 -maxdepth 3 -name "suite_summary.json" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "  runs   : ${PRETRAIN_DONE} pretrained / ${EVAL_DONE} fully done  (${TOTAL} run dirs total)"

GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
echo "  GPU    : ${GPU} (util%, mem_used MB, mem_total MB)"

DISK=$(df -h / 2>/dev/null | tail -1 | awk '{print $3"/"$2" used ("$5")"}')
echo "  disk   : $DISK"

echo "  recent :"
if [ -f "$LOG_FILE" ]; then
  grep -av "upload:\|backup_status" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/    /'
else
  echo "    (no log yet: $LOG_FILE)"
fi
REMOTE

  done

  echo ""
  echo "------------------------------------------------------------"
  echo "S3 experiment dirs (most recent first):"
  aws s3 ls s3://climb-s3-bucket/experiments/robust_matrix/ 2>/dev/null \
    | sort -r | awk '{print "  "$NF}' | head -12
  echo ""
  echo "Next poll in ${INTERVAL}s  (Ctrl-C to stop)"
  sleep "$INTERVAL"
done

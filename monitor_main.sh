#!/usr/bin/env bash
# Monitor the main experiment wave across all 3 workers.
# Usage:  bash monitor_main.sh [poll_interval_seconds]

INTERVAL="${1:-60}"
KEY="/Users/lsieben/VSCode/CLIMB/climb-gpu-key.pem"

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

# Colour helpers (no-op if terminal doesn't support)
_bold()  { printf '\033[1m%s\033[0m'  "$*"; }
_cyan()  { printf '\033[1;36m%s\033[0m' "$*"; }
_green() { printf '\033[1;32m%s\033[0m' "$*"; }
_red()   { printf '\033[1;31m%s\033[0m' "$*"; }
_dim()   { printf '\033[2m%s\033[0m'  "$*"; }

_worker_banner() {
  local worker="$1" host="$2"
  local W=74
  local tag
  case "$worker" in
    worker0) tag="W0 · unsupervised ramp-up  (3 × 1M→10B)" ;;
    worker1) tag="W1 · supervised-only ablations  (25 runs)" ;;
    worker2) tag="W2 · coverage + mixed-ratio ablations  (20 runs)" ;;
  esac
  local border
  border=$(printf '=%.0s' $(seq 1 $W))

  echo ""
  echo "  $border"
  printf "  =   %-*s   =\n" $((W-6)) ""
  printf "  =   %-*s   =\n" $((W-6)) "  $(echo "$worker" | tr 'a-z' 'A-Z')   ///   $host"
  printf "  =   %-*s   =\n" $((W-6)) "  $tag"
  printf "  =   %-*s   =\n" $((W-6)) ""
  echo "  $border"
}

_worker_block() {
  local worker="$1"
  local host
  host=$(_get_host "$worker")
  local W=74   # inner content width

  _worker_banner "$worker" "$host"

  local remote_out
  remote_out=$(ssh -i "$KEY" \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=8 \
      -o BatchMode=yes \
      "ec2-user@${host}" \
      "WORKER=$worker bash -s" << 'REMOTE' 2>/dev/null

# ── process status ─────────────────────────────────────────────────────────
PRETRAIN_PID=$(pgrep -f "pretrain_pipeline"      2>/dev/null | head -1 || true)
LAUNCHER_PID=$(pgrep -f "launch_experiment_wave" 2>/dev/null | head -1 || true)
EVAL_PID=$(pgrep    -f "run_moleculenet_suite|evaluate_model" 2>/dev/null | head -1 || true)

if   [ -n "$PRETRAIN_PID" ]; then STATUS="pretraining   (pid $PRETRAIN_PID)"
elif [ -n "$EVAL_PID"     ]; then STATUS="evaluating    (pid $EVAL_PID)"
elif [ -n "$LAUNCHER_PID" ]; then STATUS="launcher idle (pid $LAUNCHER_PID)"
else                               STATUS="NO ACTIVE PROCESS"
fi

# ── active run ────────────────────────────────────────────────────────────
ACTIVE_CONFIG=$(ps -eo args 2>/dev/null \
  | grep pretrain_pipeline | grep -v grep \
  | grep -o "\-\-config [^ ]*" | awk '{print $2}' | head -1)
ACTIVE_RUN=""
if [ -n "$ACTIVE_CONFIG" ]; then
  ACTIVE_RUN=$(echo "$ACTIVE_CONFIG" \
    | sed 's|.*/robust_matrix/||' | sed 's|/config.yaml||')
fi

# ── run counts ────────────────────────────────────────────────────────────
EXP_ROOT="/home/ec2-user/CLIMB/experiments/robust_matrix"
PRETRAIN_DONE=$(find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -name "training_results.json" 2>/dev/null | wc -l | tr -d ' ')
EVAL_DONE=$(    find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -name "suite_summary.json"    2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(        find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d                       2>/dev/null | wc -l | tr -d ' ')

# ── progress bar ──────────────────────────────────────────────────────────
METRICS_FILE="$EXP_ROOT/${ACTIVE_RUN}/metrics.jsonl"
PROGRESS_LINE="(waiting for first metrics)"
METRICS_LINE=""

if [ -n "$ACTIVE_RUN" ] && [ -f "$METRICS_FILE" ]; then
  LAST=$(tail -1 "$METRICS_FILE")

  # Find most recently modified trainer_state.json for step-based progress
  TRAINER_STATE_PATH=""
  if [ -n "$ACTIVE_RUN" ]; then
    TRAINER_STATE_PATH=$(find "$EXP_ROOT/${ACTIVE_RUN}" -name "trainer_state.json" \
      2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  fi

  PARSED=$(python3 - "$LAST" "$TRAINER_STATE_PATH" << 'PYEOF'
import json, sys

raw              = sys.argv[1]
trainer_path     = sys.argv[2] if len(sys.argv) > 2 else ""
BAR = 30

try:
    d     = json.loads(raw)
    ts    = d.get("tokens_seen") or 0
    tb    = d.get("token_budget")
    step  = d.get("global_step", 0)
    loss  = d.get("loss")
    lr    = d.get("learning_rate")
    phase = str(d.get("phase") or "unknown")
    elapsed = d.get("elapsed_seconds") or 0

    # 1) Token-budget progress (unsupervised / mixed runs)
    if tb and tb > 0:
        pct    = min(100, int(ts / tb * 100))
        filled = int(BAR * pct / 100)
        bar    = "#" * filled + "." * (BAR - filled)
        prog   = "[%s] %3d%%  %.1fM / %.0fM tokens" % (bar, pct, ts/1e6, tb/1e6)

    # 2) Step-based progress from HF trainer_state.json (supervised runs)
    else:
        max_steps = 0
        cur_step  = step
        try:
            with open(trainer_path) as f:
                ts_data = json.load(f)
            max_steps = ts_data.get("max_steps", 0)
            cur_step  = ts_data.get("global_step", step)
        except Exception:
            pass

        if max_steps > 0:
            pct    = min(100, int(cur_step / max_steps * 100))
            filled = int(BAR * pct / 100)
            bar    = "#" * filled + "." * (BAR - filled)
            if elapsed and cur_step > 0:
                secs_per_step = elapsed / cur_step
                remaining = int(secs_per_step * (max_steps - cur_step))
                h, m = divmod(remaining // 60, 60)
                eta_s = "%dh%02dm left" % (h, m)
            else:
                eta_s = "ETA unknown"
            fam = phase.replace("supervised_","").replace("unsupervised","unsup")
            prog = "[%s] %3d%%  step %d/%d  [%s]  %s" % (bar, pct, cur_step, max_steps, fam, eta_s)
        else:
            fam  = phase.replace("supervised_","").replace("unsupervised","unsup")
            prog = "[%s] step %d  [%s]" % ("." * BAR, step, fam)

    loss_s   = ("%.4g" % loss) if isinstance(loss, float) else "n/a"
    lr_s     = ("%.2e" % lr)   if isinstance(lr,   float) else "n/a"
    fam_disp = phase.replace("supervised_","sup/")
    met      = "loss=%-12s  lr=%s  step=%-8d  phase=%s" % (loss_s, lr_s, step, fam_disp)
    print(prog + "|||" + met)
except Exception as e:
    print("(parse error: %s)|||" % e)
PYEOF
)
  PROGRESS_LINE=$(echo "$PARSED" | cut -d'|' -f1)
  METRICS_LINE=$(  echo "$PARSED" | awk -F'|||' '{print $2}')
fi

# ── GPU & disk ─────────────────────────────────────────────────────────────
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 \
  | awk -F', ' '{printf "%s%% util  %s / %s MB", $1, $2, $3}' || echo "n/a")
DISK=$(df -h / 2>/dev/null | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')

# ── recent log — prefer nohup_launcher.log (active), fall back to wave log ──
NOHUP_LOG="/home/ec2-user/nohup_launcher.log"
WAVE_LOG="/home/ec2-user/artifacts/robust_matrix_logs/main_wave_${WORKER}.log"
if [ -f "$NOHUP_LOG" ]; then
  RECENT=$(grep -av "upload:\|backup_status\|it/s\|\[A\|━\|▏\|▎\|▍\|▌\|▋\|▊\|▉\|█\|Traceback\|File \"/\|  raise \|subprocess\." \
    "$NOHUP_LOG" | tail -4)
elif [ -f "$WAVE_LOG" ]; then
  RECENT=$(grep -av "upload:\|backup_status\|it/s\|\[A\|━\|▏\|▎\|▍\|▌\|▋\|▊\|▉\|█\|Traceback\|File \"/\|  raise \|subprocess\." \
    "$WAVE_LOG" | tail -4)
else
  RECENT="(no log yet)"
fi

# ── emit fields ────────────────────────────────────────────────────────────
printf 'STATUS=%s\n'   "$STATUS"
printf 'RUN=%s\n'      "${ACTIVE_RUN:-idle}"
printf 'PROGRESS=%s\n' "$PROGRESS_LINE"
printf 'METRICS=%s\n'  "$METRICS_LINE"
printf 'DONE=%s/%s/%s\n' "$PRETRAIN_DONE" "$EVAL_DONE" "$TOTAL"
printf 'GPU=%s\n'      "$GPU"
printf 'DISK=%s\n'     "$DISK"
printf 'LOG1=%s\n'     "$(echo "$RECENT" | sed -n '1p')"
printf 'LOG2=%s\n'     "$(echo "$RECENT" | sed -n '2p')"
printf 'LOG3=%s\n'     "$(echo "$RECENT" | sed -n '3p')"
printf 'LOG4=%s\n'     "$(echo "$RECENT" | sed -n '4p')"
REMOTE
  )

  # ── parse fields from remote output ────────────────────────────────────
  local STATUS RUN PROGRESS METRICS DONE GPU DISK LOG1 LOG2 LOG3 LOG4
  if [ -z "$remote_out" ]; then
    STATUS="SSH UNREACHABLE"
    RUN="" PROGRESS="" METRICS="" DONE="?/?/?" GPU="n/a" DISK="n/a"
    LOG1="" LOG2="" LOG3="" LOG4=""
  else
    STATUS=$(  echo "$remote_out" | grep '^STATUS='   | head -1 | cut -d= -f2-)
    RUN=$(     echo "$remote_out" | grep '^RUN='      | head -1 | cut -d= -f2-)
    PROGRESS=$(echo "$remote_out" | grep '^PROGRESS=' | head -1 | cut -d= -f2-)
    METRICS=$( echo "$remote_out" | grep '^METRICS='  | head -1 | cut -d= -f2-)
    DONE=$(    echo "$remote_out" | grep '^DONE='     | head -1 | cut -d= -f2-)
    GPU=$(     echo "$remote_out" | grep '^GPU='      | head -1 | cut -d= -f2-)
    DISK=$(    echo "$remote_out" | grep '^DISK='     | head -1 | cut -d= -f2-)
    LOG1=$(    echo "$remote_out" | grep '^LOG1='     | head -1 | cut -d= -f2-)
    LOG2=$(    echo "$remote_out" | grep '^LOG2='     | head -1 | cut -d= -f2-)
    LOG3=$(    echo "$remote_out" | grep '^LOG3='     | head -1 | cut -d= -f2-)
    LOG4=$(    echo "$remote_out" | grep '^LOG4='     | head -1 | cut -d= -f2-)
  fi

  local PRETRAIN_DONE EVAL_DONE TOTAL
  PRETRAIN_DONE=$(echo "$DONE" | cut -d/ -f1)
  EVAL_DONE=$(    echo "$DONE" | cut -d/ -f2)
  TOTAL=$(        echo "$DONE" | cut -d/ -f3)

  # ── print body ─────────────────────────────────────────────────────────
  local INNER SEP
  INNER=$((W-2))
  SEP=$(printf -- '-%.0s' $(seq 1 $W))

  printf "  | %-*s |\n" $INNER "  status  : ${STATUS}"
  printf "  | %-*s |\n" $INNER "  run     : ${RUN:-idle}"
  printf "  | %-*s |\n" $INNER "  $SEP"
  printf "  | %-*s |\n" $INNER "  progress: ${PROGRESS:-(waiting for metrics)}"
  printf "  | %-*s |\n" $INNER "  metrics : ${METRICS}"
  printf "  | %-*s |\n" $INNER "  $SEP"
  printf "  | %-*s |\n" $INNER "  done    : ${PRETRAIN_DONE} pretrained / ${EVAL_DONE} evaluated  (${TOTAL} run dirs)"
  printf "  | %-*s |\n" $INNER "  GPU     : ${GPU}"
  printf "  | %-*s |\n" $INNER "  disk    : ${DISK}"
  printf "  | %-*s |\n" $INNER "  $SEP"
  printf "  | %-*s |\n" $INNER "  log (nohup_launcher.log):"
  [ -n "$LOG1" ] && printf "  |   %-*s |\n" $((INNER-2)) "${LOG1:0:$((INNER-2))}"
  [ -n "$LOG2" ] && printf "  |   %-*s |\n" $((INNER-2)) "${LOG2:0:$((INNER-2))}"
  [ -n "$LOG3" ] && printf "  |   %-*s |\n" $((INNER-2)) "${LOG3:0:$((INNER-2))}"
  [ -n "$LOG4" ] && printf "  |   %-*s |\n" $((INNER-2)) "${LOG4:0:$((INNER-2))}"

  local BORDER
  BORDER=$(printf '=%.0s' $(seq 1 $W))
  echo "  $BORDER"
}

# ── main loop ──────────────────────────────────────────────────────────────
while true; do
  clear

  echo ""
  echo "   _____ _     ___ __  __ ____    __  __             _ _"
  echo "  / ____| |   |_ _|  \/  | __ )  |  \/  | ___  _ __ (_) |_ ___  _ __"
  echo " | |    | |    | || |\/| |  _ \  | |\/| |/ _ \| '_ \| | __/ _ \| '__|"
  echo " | |___ | |___ | || |  | | |_) | | |  | | (_) | | | | | || (_) | |"
  echo "  \____|_____|___|_|  |_|____/  |_|  |_|\___/|_| |_|_|\__\___/|_|"
  echo ""
  printf "  Refreshed: %s   poll=%ss   Ctrl-C to stop\n" \
    "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$INTERVAL"
  echo ""

  for worker in worker0 worker1 worker2; do
    _worker_block "$worker"
    echo ""
    echo ""
  done

  echo "  S3 experiment dirs (most recent first):"
  aws s3 ls s3://climb-s3-bucket/experiments/robust_matrix/ 2>/dev/null \
    | sort -r | awk '{print "    " $NF}' | head -12

  echo ""
  printf "  Next refresh in %ss ...\n" "$INTERVAL"
  sleep "$INTERVAL"
done

#!/usr/bin/env bash
# Monitor the main experiment wave across all 5 workers.
# Usage:  bash monitor_main.sh [poll_interval_seconds]
# Compatible with bash 3.2+ (macOS default).

INTERVAL="${1:-60}"
KEY="/Users/lsieben/VSCode/CLIMB/climb-gpu-key.pem"

trap 'echo; echo "Stopped."; exit 0' INT TERM

_get_host() {
  case "$1" in
    worker0) echo "3.89.64.232" ;;
    worker1) echo "54.89.239.106" ;;
    worker2) echo "18.206.240.192" ;;
    worker3) echo "13.216.245.191" ;;
    worker4) echo "13.221.44.139" ;;
  esac
}

_get_role() {
  case "$1" in
    worker0) echo "unsup 10B_00  →  sup_order_1of5 ×5" ;;
    worker1) echo "sup_order 1of5 ×5  +  2of5 ×5" ;;
    worker2) echo "coverage + mixed-ratio ablations ×20" ;;
    worker3) echo "unsup 10B_01 + 10B_02" ;;
    worker4) echo "sup_order 3of5/4of5/5of5 ×15" ;;
  esac
}

_bold()  { printf '\033[1m%s\033[0m'  "$*"; }
_green() { printf '\033[1;32m%s\033[0m' "$*"; }
_red()   { printf '\033[1;31m%s\033[0m' "$*"; }
_yellow(){ printf '\033[1;33m%s\033[0m' "$*"; }
_dim()   { printf '\033[2m%s\033[0m'  "$*"; }

_worker_line() {
  local worker="$1"
  local host
  host=$(_get_host "$worker")

  local remote_out
  remote_out=$(ssh -i "$KEY" \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=8 \
      -o BatchMode=yes \
      "ec2-user@${host}" bash << 'REMOTE' 2>/dev/null

EXP_ROOT="/home/ec2-user/CLIMB/experiments/robust_matrix"

PRETRAIN_PID=$(pgrep -f "pretrain_pipeline"      2>/dev/null | head -1 || true)
LAUNCHER_PID=$(pgrep -f "launch_experiment_wave" 2>/dev/null | head -1 || true)
EVAL_PID=$(pgrep    -f "run_moleculenet_suite|evaluate_model" 2>/dev/null | head -1 || true)

if   [ -n "$PRETRAIN_PID" ]; then STATUS="train"
elif [ -n "$EVAL_PID"     ]; then STATUS="eval"
elif [ -n "$LAUNCHER_PID" ]; then STATUS="idle"
else                               STATUS="STOPPED"
fi

ACTIVE_CONFIG=$(ps -eo args 2>/dev/null \
  | grep pretrain_pipeline | grep -v grep \
  | grep -o "\-\-config [^ ]*" | awk '{print $2}' | head -1)
ACTIVE_RUN=""
if [ -n "$ACTIVE_CONFIG" ]; then
  ACTIVE_RUN=$(echo "$ACTIVE_CONFIG" | sed 's|.*/robust_matrix/||' | sed 's|/config.yaml||')
fi

PRETRAIN_DONE=$(find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -name "training_results.json" 2>/dev/null | wc -l | tr -d ' ')
EVAL_DONE=$(    find "$EXP_ROOT" -mindepth 2 -maxdepth 2 -name "suite_summary.json"    2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(        find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d                       2>/dev/null | wc -l | tr -d ' ')

PROGRESS=""
METRICS_FILE="$EXP_ROOT/${ACTIVE_RUN}/metrics.jsonl"
TRAINER_STATE=$(find "$EXP_ROOT/${ACTIVE_RUN}" -name "trainer_state.json" \
  2>/dev/null | xargs ls -t 2>/dev/null | head -1)

if [ -n "$ACTIVE_RUN" ] && [ -s "$METRICS_FILE" ]; then
  LAST=$(tail -1 "$METRICS_FILE")
  PROGRESS=$(python3 << PYEOF
import json, sys
BAR = 24
try:
    raw          = """$LAST"""
    trainer_path = """$TRAINER_STATE"""
    d     = json.loads(raw)
    ts    = d.get("tokens_seen") or 0
    tb    = d.get("token_budget")
    step  = d.get("global_step", 0)
    elapsed = d.get("elapsed_seconds") or 0

    if tb and tb > 0:
        pct   = min(100, int(ts / tb * 100))
        filled = int(BAR * pct / 100)
        bar   = "#" * filled + "." * (BAR - filled)
        rate  = ts / elapsed if elapsed > 0 else 0
        eta_s = ""
        if rate > 0:
            rem = int((tb - ts) / rate)
            h, m = divmod(rem // 60, 60)
            eta_s = "  ETA %dh%02dm" % (h, m)
        print("[%s] %3d%%  %.2fB/%.0fB tok%s" % (bar, pct, ts/1e9, tb/1e9, eta_s))
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
            pct   = min(100, int(cur_step / max_steps * 100))
            filled = int(BAR * pct / 100)
            bar   = "#" * filled + "." * (BAR - filled)
            eta_s = ""
            if elapsed > 0 and cur_step > 0:
                rem = int(elapsed / cur_step * (max_steps - cur_step))
                h, m = divmod(rem // 60, 60)
                eta_s = "  ETA %dh%02dm" % (h, m)
            print("[%s] %3d%%  step %d/%d%s" % (bar, pct, cur_step, max_steps, eta_s))
        else:
            print("[%s] step %d  (no max_steps yet)" % ("." * BAR, step))
except Exception as e:
    print("(parse error: %s)" % e)
PYEOF
)
elif [ -n "$TRAINER_STATE" ] && [ -f "$TRAINER_STATE" ]; then
  PROGRESS=$(python3 << PYEOF
import json, sys
BAR = 24
try:
    with open("""$TRAINER_STATE""") as f:
        d = json.load(f)
    cur  = d.get("global_step", 0)
    maxs = d.get("max_steps", 0)
    if maxs > 0:
        pct   = min(100, int(cur / maxs * 100))
        filled = int(BAR * pct / 100)
        bar   = "#" * filled + "." * (BAR - filled)
        print("[%s] %3d%%  step %d/%d" % (bar, pct, cur, maxs))
    elif cur > 0:
        print("[%s] step %d  (warming up)" % ("." * BAR, cur))
except Exception as e:
    print("(trainer_state error: %s)" % e)
PYEOF
)
fi

NOHUP_LOG="/home/ec2-user/nohup_launcher.log"
LAST_LOG=""
if [ -f "$NOHUP_LOG" ]; then
  LAST_LOG=$(grep -av "upload:\|backup_status\|it/s\|\[A\|━\|▏\|▎\|▍\|▌\|▋\|▊\|▉\|█\|Traceback\|File \"/\|  raise \|subprocess\." \
    "$NOHUP_LOG" | tail -1)
fi

printf 'STATUS=%s\n' "$STATUS"
printf 'RUN=%s\n'    "${ACTIVE_RUN:-idle}"
printf 'DONE=%s/%s\n' "$PRETRAIN_DONE" "$EVAL_DONE"
printf 'TOTAL=%s\n'  "$TOTAL"
printf 'PROG=%s\n'   "$PROGRESS"
printf 'LOG=%s\n'    "$LAST_LOG"
REMOTE
  )

  local STATUS RUN DONE TOTAL PROG LOG
  if [ -z "$remote_out" ]; then
    STATUS="UNREACHABLE"; RUN=""; DONE="?/?"; TOTAL="?"; PROG=""; LOG=""
  else
    STATUS=$(echo "$remote_out" | grep '^STATUS=' | head -1 | cut -d= -f2-)
    RUN=$(   echo "$remote_out" | grep '^RUN='    | head -1 | cut -d= -f2-)
    DONE=$(  echo "$remote_out" | grep '^DONE='   | head -1 | cut -d= -f2-)
    TOTAL=$( echo "$remote_out" | grep '^TOTAL='  | head -1 | cut -d= -f2-)
    PROG=$(  echo "$remote_out" | grep '^PROG='   | head -1 | cut -d= -f2-)
    LOG=$(   echo "$remote_out" | grep '^LOG='    | head -1 | cut -d= -f2-)
  fi

  local PRETRAIN_DONE EVAL_DONE
  PRETRAIN_DONE=$(echo "$DONE" | cut -d/ -f1)
  EVAL_DONE=$(    echo "$DONE" | cut -d/ -f2)

  local STATUS_FMT
  case "$STATUS" in
    train)       STATUS_FMT=$(_green "training") ;;
    eval)        STATUS_FMT=$(_green "evaluating") ;;
    idle)        STATUS_FMT=$(_yellow "launcher idle") ;;
    STOPPED)     STATUS_FMT=$(_red "STOPPED") ;;
    UNREACHABLE) STATUS_FMT=$(_red "UNREACHABLE") ;;
    *)           STATUS_FMT="$STATUS" ;;
  esac

  local WNUM="${worker/worker/W}"
  printf "  $(_bold "$WNUM")  [%-15s]  %s\n" "$STATUS_FMT" "$(_get_role "$worker")"
  printf "      run    : %s\n"   "${RUN:-idle}"
  printf "      done   : %s pretrained  /  %s evaluated  (%s dirs)\n" \
    "$PRETRAIN_DONE" "$EVAL_DONE" "$TOTAL"
  [ -n "$PROG" ] && printf "      progress: %s\n" "$PROG"
  [ -n "$LOG"  ] && printf "      $(_dim "log    : %s")\n" "${LOG:0:100}"
  echo ""
}

while true; do
  clear
  echo ""
  printf "  $(_bold "CLIMB Experiment Monitor")   %s   (poll=%ss)\n" \
    "$(date -u '+%Y-%m-%d %H:%M UTC')" "$INTERVAL"
  echo "  ─────────────────────────────────────────────────────────────────────"
  echo ""

  for worker in worker0 worker1 worker2 worker3 worker4; do
    _worker_line "$worker"
  done

  echo "  ─────────────────────────────────────────────────────────────────────"
  echo "  S3 (recent):"
  aws s3 ls s3://climb-s3-bucket/experiments/robust_matrix/ 2>/dev/null \
    | sort -r | awk '{print "    " $NF}' | head -8
  echo ""
  printf "  Next refresh in %ss ...\n" "$INTERVAL"
  sleep "$INTERVAL"
done

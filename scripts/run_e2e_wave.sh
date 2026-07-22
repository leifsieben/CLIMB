#!/usr/bin/env bash
# On-box driver for the `no_pretrain_end_to_end` arm (Fig A1's missing bar).
#
# Runs scripts/run_e2e_random.py for one or more replicates behind a STALL watchdog.
# The watchdog reads the run's heartbeat (samples_seen, written after every epoch) and
# only kills a job whose progress counter has stopped advancing -- never one that is
# merely slow. A slow fine-tune is a config bug worth diagnosing; killing it destroys
# hours of real work and tells you nothing.
#
# Deliberately does NOT stop the instance on exit: this box also hosts an unrelated
# long-running CPU job, so shutting it down to save GPU cost would silently kill that.
# Idle cost is handled by the operator, not by this script.
#
#   Usage: run_e2e_wave.sh <tag> <replicate> [<replicate> ...]
#   Env:   STALL_MIN (default 45), EPOCHS, S3ROOT, OUTROOT
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python

TAG=${1:?tag required}; shift
REPS=("$@")
[ ${#REPS[@]} -gt 0 ] || { echo "no replicates given"; exit 2; }

STALL_MIN=${STALL_MIN:-45}
EPOCHS=${EPOCHS:-20}
OUTROOT=${OUTROOT:-experiments/_e2e/out}
S3ROOT=${S3ROOT:-s3://climb-s3-bucket/experiments/climb_v2_phase2}
TOK=experiments/_e2e/_tokenizer

mkdir -p "$OUTROOT"
say(){ echo "[e2e $(date -u +%H:%M:%S)] $*" | tee -a "e2e_${TAG}.log"; }

bash scripts/notify.sh INFO "e2e_random START ($TAG: reps ${REPS[*]})" \
  "no_pretrain_end_to_end arm; 6 tasks x {hold-out, 5-fold CV}; stall limit ${STALL_MIN}m"

overall=0
for R in "${REPS[@]}"; do
  RID=$(printf "e2e_random_%02d" "$R")
  RUNDIR="$OUTROOT/$RID"
  LOG="e2e_${TAG}_${RID}.log"
  mkdir -p "$RUNDIR"
  # A verified run is already done. Skip logic reads verified.json -- the SAME source of
  # truth the runner writes only after checking achieved work -- never file existence.
  if [ -f "$RUNDIR/verified.json" ]; then say "$RID already verified; skipping"; continue; fi

  say "launching $RID (encoder experiments/_e2e/enc_$(printf %02d "$R"))"
  setsid "$PY" scripts/run_e2e_random.py \
      --replicate "$R" \
      --encoder "experiments/_e2e/enc_$(printf %02d "$R")" \
      --tokenizer "$TOK" \
      --output_root "$OUTROOT" \
      --epochs "$EPOCHS" \
      --s3 "$S3ROOT" \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  sleep 5
  say "$RID pid=$PID ppid=$(ps -o ppid= -p $PID 2>/dev/null | tr -d ' ')"

  HB="$RUNDIR/heartbeat.json"
  last_seen=-1; last_change=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    seen=$($PY -c "import json,sys;print(json.load(open('$HB'))['samples_seen'])" 2>/dev/null || echo -1)
    if [ "$seen" != "$last_seen" ]; then last_seen=$seen; last_change=$now; fi
    stalled=$(( (now - last_change) / 60 ))
    if [ "$stalled" -ge "$STALL_MIN" ]; then
      say "STALL: $RID progress counter frozen at $seen for ${stalled}m -- killing"
      bash scripts/notify.sh ALERT "e2e_random STALL ($RID)" \
        "samples_seen stuck at $seen for ${stalled}m; killing pid $PID. Partial results in $S3ROOT/$RID"
      kill -TERM "$PID" 2>/dev/null; sleep 20; kill -KILL "$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null; rc=$?
  say "$RID exited rc=$rc (samples_seen=$last_seen)"

  # Save BEFORE judging: whatever happened, the partial work belongs in S3.
  aws s3 sync "$RUNDIR" "$S3ROOT/$RID" --only-show-errors
  aws s3 cp "$LOG" "$S3ROOT/$RID/_logs/$LOG" --only-show-errors

  if [ -f "$RUNDIR/verified.json" ]; then
    bash scripts/notify.sh DONE "e2e_random $RID VERIFIED" "$(cat "$RUNDIR/verification_report.json" | head -c 900)"
  else
    overall=1
    bash scripts/notify.sh ALERT "e2e_random $RID NOT VERIFIED (rc=$rc)" \
      "$(cat "$RUNDIR/verification_report.json" 2>/dev/null | head -c 900)"
  fi
done

say "wave $TAG finished overall=$overall"
echo "E2E_WAVE_DONE tag=$TAG rc=$overall"
exit "$overall"

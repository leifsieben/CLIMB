#!/bin/bash
# B1p1 label-efficiency, the `no_pretrain_end_to_end` series -- the one arm b1_replicates_v2.sh
# could not produce.
#
# b1_replicates_v2.sh sweeps four series that are all FROZEN probes: one encoder forward pass
# per molecule, then an MLP re-fit at each label budget. Its own closing comment says the
# end-to-end arm "needs a fine-tuned sweep, not this frozen-probe path" -- so B1p1's legend has
# carried it as "NOT RUN". This script is that fine-tuned sweep. Same grid, same encoder, same
# subsample draws; the encoder is unfrozen instead of probed.
#
# Structure mirrors run_e2e_wave.sh rather than b1_replicates_v2.sh: the frozen cells take
# seconds and can be a bare loop, but a fine-tuned cell runs for many minutes on the GPU and
# needs a stall watchdog, provable completion, and incremental S3 saves.
#
# The watchdog kills only a job whose heartbeat counter has FROZEN. A merely slow fine-tune is
# a config bug to diagnose, not a hang; killing it destroys real work and explains nothing.
#
# Does NOT stop the instance. worker3 also hosts an unrelated multi-hour CPU job, and a script
# that shuts a box down to save GPU cost would silently kill it. Stopping is the operator's call.
#
#   Usage: b1_e2e_v2.sh <tag> <n:seed> [<n:seed> ...]      e.g.  b1_e2e_v2.sh w5 100:0 300:0
#          n=0 means the full training set (b1_replicates_v2.sh's encoding).
#   Env:   STALL_MIN (default 45), EPOCHS, TASKS, S3ROOT, OUT
set -uo pipefail
cd /home/ec2-user/CLIMB 2>/dev/null || cd /Users/lsieben/VSCode/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}

TAG=${1:?tag required}; shift
CELLS=("$@")
[ ${#CELLS[@]} -gt 0 ] || { echo "no cells given (expect n:seed)"; exit 2; }

STALL_MIN=${STALL_MIN:-45}
EPOCHS=${EPOCHS:-20}
OUT=${OUT:-figure_data/climb_v2_labeleff_v2}
S3ROOT=${S3ROOT:-s3://climb-s3-bucket/experiments/climb_v2_labeleff_v2}
TASKS=${TASKS:-"ESOL Lipophilicity QM7 BBBP BACE Tox21 HIV"}

# Requirement 1 of the panel: the SAME weights B1p1's `random` series froze. Staged from S3 so
# every box provably uses one artifact rather than whatever happens to be on its disk.
ENC=experiments/_b1e2e/encoder
TOK=experiments/_e2e/_tokenizer
mkdir -p "$OUT" experiments/_b1e2e
[ -f "$ENC/model.safetensors" ] || \
  aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2/random_baseline_00/encoder "$ENC" --only-show-errors
[ -f "$TOK/tokenizer.json" ] || \
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors

say(){ echo "[b1e2e $(date -u +%H:%M:%S)] $*" | tee -a "b1e2e_${TAG}.log"; }

bash scripts/notify.sh INFO "b1_e2e START ($TAG: ${CELLS[*]})" \
  "Fig B1p1 no_pretrain_end_to_end; 7 tasks x 3 fine-tune seeds per cell; stall limit ${STALL_MIN}m"

overall=0
for C in "${CELLS[@]}"; do
  N=${C%%:*}; S=${C##*:}
  CELL="e2e_n${N}_s${S}"
  CELLDIR="$OUT/$CELL"
  LOG="b1e2e_${TAG}_${CELL}.log"
  mkdir -p "$CELLDIR"
  # Skip logic reads verified.json -- the SAME source of truth the runner writes only after
  # checking achieved work. Never the summary CSV, which is rewritten after every dataset and
  # so exists in a well-formed but incomplete state for most of each cell's life.
  if [ -f "$CELLDIR/verified.json" ]; then say "$CELL already verified; skipping"; continue; fi

  say "launching $CELL (budget=$N subsample_seed=$S)"
  setsid "$PY" scripts/run_b1_e2e_cell.py \
      --n "$N" --seed "$S" \
      --encoder "$ENC" --tokenizer "$TOK" \
      --out_root "$OUT" --tasks $TASKS \
      --head_seeds 0 1 2 --epochs "$EPOCHS" \
      --s3 "$S3ROOT" --heartbeat "$CELLDIR/heartbeat.json" \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  sleep 5
  say "$CELL pid=$PID ppid=$(ps -o ppid= -p $PID 2>/dev/null | tr -d ' ')"

  HB="$CELLDIR/heartbeat.json"
  last_seen=-1; last_change=$(date +%s); tick=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    now=$(date +%s)
    seen=$("$PY" -c "import json;print(json.load(open('$HB'))['samples_seen'])" 2>/dev/null || echo -1)
    if [ "$seen" != "$last_seen" ]; then last_seen=$seen; last_change=$now; fi
    # Push partial work to S3 DURING the cell, not only when it ends. run_b1_e2e_cell.py
    # rewrites the summary after every dataset, so a cell killed on HIV still has six tasks
    # worth of finished fine-tuning on disk -- which is worth nothing if it never leaves the
    # box. The full-label cell runs ~90 min; losing that to an end-only upload is the exact
    # failure we just watched happen to the E1 sup_only job.
    tick=$((tick+1))
    if [ $((tick % 10)) -eq 0 ]; then
      aws s3 sync "$CELLDIR" "$S3ROOT/$CELL" --only-show-errors || true
    fi
    stalled=$(( (now - last_change) / 60 ))
    if [ "$stalled" -ge "$STALL_MIN" ]; then
      say "STALL: $CELL counter frozen at $seen for ${stalled}m -- killing"
      bash scripts/notify.sh ALERT "b1_e2e STALL ($CELL)" \
        "samples_seen stuck at $seen for ${stalled}m; killing pid $PID. Partial work in $S3ROOT/$CELL"
      kill -TERM "$PID" 2>/dev/null; sleep 20; kill -KILL "$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null; rc=$?
  say "$CELL exited rc=$rc (samples_seen=$last_seen)"

  # Save BEFORE judging: whatever happened, the partial work belongs in S3.
  aws s3 sync "$CELLDIR" "$S3ROOT/$CELL" --only-show-errors
  aws s3 cp "$LOG" "$S3ROOT/$CELL/_logs/$LOG" --only-show-errors

  if [ -f "$CELLDIR/verified.json" ]; then
    say "$CELL VERIFIED"
  else
    overall=1
    bash scripts/notify.sh ALERT "b1_e2e $CELL NOT VERIFIED (rc=$rc)" \
      "$(cat "$CELLDIR/verification_report.json" 2>/dev/null | head -c 900)"
  fi
done

say "wave $TAG finished overall=$overall"
bash scripts/notify.sh DONE "b1_e2e wave $TAG finished (rc=$overall)" "cells: ${CELLS[*]}"
echo "B1_E2E_WAVE_DONE tag=$TAG rc=$overall"
exit "$overall"

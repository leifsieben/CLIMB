#!/usr/bin/env bash
# Phase-2 per-worker runner (hardened). Runs ONE pre-split worker manifest with:
#   - a PREFLIGHT gate (refuses to launch into a bad environment)
#   - a sidecar S3 sync (spot-reclaim safe)
#   - a HEARTBEAT + stall-alert sidecar that emails progress directly (no Claude)
#   - a completion GATE: the box only self-stops if EVERY run in its manifest is
#     verified complete (reached its FP budget). Otherwise it ALERTS and STAYS UP
#     so an incomplete run can be inspected/resumed — it never again shuts down on
#     truncated work and hides the failure.
#
# Usage (on the box, detached):
#   nohup bash scripts/phase2_worker.sh <worker_manifest.json> <worker_name> &
set -x
cd /home/ec2-user/CLIMB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export CLIMB_SNS_ARN="${CLIMB_SNS_ARN:-arn:aws:sns:us-east-1:075120018132:climb-experiments}"
export CLIMB_SNS_REGION="${CLIMB_SNS_REGION:-us-east-1}"
PY=/home/ec2-user/venvs/climb/bin/python
MANIFEST="${1:?usage: phase2_worker.sh <manifest> <worker_name>}"
WORKER="${2:?usage: phase2_worker.sh <manifest> <worker_name>}"
export CLIMB_WORKER="$WORKER"
# Wave roots come from the manifest (`results_root` / `s3_backup_root`) so this runner is not
# pinned to climb_v2_phase2 -- the lrsweep/ablation waves live under their own prefixes, and a
# hardcoded root would sync their runs into the wrong tree. Falls back to the phase-2 values so
# older manifests that predate those keys still work.
LOCAL=$($PY -c "
import json
m = json.load(open('$MANIFEST'))
print(m.get('results_root', 'experiments/climb_v2_phase2') if isinstance(m, dict) else 'experiments/climb_v2_phase2')
")
S3=$($PY -c "
import json
m = json.load(open('$MANIFEST'))
print(m.get('s3_backup_root', 's3://climb-s3-bucket/experiments/climb_v2_phase2') if isinstance(m, dict) else 's3://climb-s3-bucket/experiments/climb_v2_phase2')
")
echo "WAVE ROOTS: local=$LOCAL s3=$S3"
NOTIFY="bash scripts/notify.sh"

# ---- PREFLIGHT: refuse to launch into a broken environment (stay up on failure) ----
if ! bash scripts/preflight.sh "$WORKER" "$MANIFEST"; then
  echo "PREFLIGHT FAILED — not launching, box staying up for inspection $(date -u)"
  exit 1   # deliberately NO shutdown: leave the box alive so it can be fixed
fi

# shared descriptor stats (identical normalization across every box); harmless if absent
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json >/dev/null 2>&1 || true

# ---- OWNERSHIP: this box may only ever WRITE the runs in its own manifest ----
# Previously every box downloaded the ENTIRE wave tree at startup and pushed the ENTIRE tree
# back every 10 min. A box therefore held copies of runs owned by other boxes, and re-uploaded
# its stale copy on every cycle -- silently reverting completed work in S3 (metrics.jsonl and
# run_status.json differ in SIZE from the good copy, so `aws s3 sync` always pushed them;
# encoders are same-size-and-older, which is the only reason model weights survived). Worse,
# the startup download then propagated the clobbered files back DOWN onto the box that had
# produced the good copy, destroying the last good original. Five completed runs were reverted
# this way before it was caught.
#
# Fix: a run has exactly one owner -- the box whose manifest lists it. Uploads are scoped to
# owned runs. Downloads pull owned runs in full (needed to resume after a restart) but pull
# only encoder/ for everything else (u2s stage-2 warm-start bases), so a run this box does not
# own can never be written by it, in either direction.
# NB: ownership keys off output_dir, NOT run_id. In the seed manifests run_id is the BASE name
# ("unsup_8M") while the actual on-disk/S3 location carries the seed suffix ("unsup_8M_s1") --
# scoping by run_id would make a seed box push its s1 results straight over the ORIGINAL
# completed unsup_8M run, i.e. re-create the very corruption this change exists to prevent.
OWN_RUNS=$($PY -c "
import json, os
m = json.load(open('$MANIFEST'))
runs = m['runs'] if isinstance(m, dict) and 'runs' in m else m
print(' '.join(os.path.basename(r['output_dir'].rstrip('/')) for r in runs))
")
echo "OWNED RUNS ($WORKER): $OWN_RUNS"

# Warm-start bases: pull exactly the encoders the manifest names in `init_encoder_path`, rather
# than sweeping every encoder under the wave prefix. Stage-2 runs frequently warm-start from a
# DIFFERENT wave (the lrsweep sweeps SFT learning rate on top of a phase-2 MLM encoder), which a
# prefix sweep would silently miss -- and a missing warm-start base kills the run on startup with
# no metrics written at all, which is exactly how all 8 original lrsweep runs died in ~15s.
INIT_ENCODERS=$($PY -c "
import json
m = json.load(open('$MANIFEST'))
runs = m['runs'] if isinstance(m, dict) and 'runs' in m else m
seen = []
for r in runs:
    for sel in (r.get('selection') or {}, (r.get('pretrain_config') or {}).get('selection') or {}):
        p = sel.get('init_encoder_path')
        if p and p not in seen:
            seen.append(p)
print(' '.join(seen))
")
for e in $INIT_ENCODERS; do
  echo "warm-start base: $e"
  mkdir -p "$e"
  aws s3 sync "s3://climb-s3-bucket/$e" "$e" >/dev/null 2>&1 || true
  if [ ! -f "$e/model.safetensors" ] && [ ! -f "$e/pytorch_model.bin" ]; then
    $NOTIFY ALERT "$WORKER MISSING WARM-START BASE" "init_encoder_path '$e' has no weights after sync. Runs depending on it would die instantly. NOT launching."
    echo "FATAL: warm-start base $e has no weights — refusing to launch $(date -u)"
    exit 1
  fi
done
# owned runs in full, so an interrupted run can resume from its checkpoint
for r in $OWN_RUNS; do
  aws s3 sync "$S3/$r" "$LOCAL/$r" --exclude "*/tokenizer/*" >/dev/null 2>&1 || true
done

# push ONLY owned runs; never the whole tree
push_owned() {
  for r in $OWN_RUNS; do
    [ -d "$LOCAL/$r" ] || continue
    aws s3 sync "$LOCAL/$r" "$S3/$r" --exclude "*/tokenizer/*" >/dev/null 2>&1
  done
}

# ---- sidecar 1: push results (incl. periodic encoder checkpoints) to S3 every 10 min ----
( while true; do
    push_owned
    sleep 600
  done ) &
SIDECAR=$!

# ---- sidecar 2: heartbeat + stall alert (direct email, no Claude in the loop) ----
( TICK=0
  while true; do
    sleep 1800   # 30-min cadence; emails a heartbeat every 4th tick (~2h)
    TICK=$((TICK+1))
    RUN=$(ls -td "$LOCAL"/*/ 2>/dev/null | head -1 | xargs -n1 basename 2>/dev/null)
    M="$LOCAL/$RUN/metrics.jsonl"
    FP=$(tail -1 "$M" 2>/dev/null | $PY -c "import sys,json;print(json.loads(sys.stdin.read()).get('forward_passes_seen',0))" 2>/dev/null || echo 0)
    MT=$(stat -c %Y "$M" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    TRAINING=$(pgrep -f "[p]retrain_v2" | head -1)
    # stall alert: a training proc exists but metrics file hasn't advanced in >25 min
    if [ -n "$TRAINING" ] && [ "$MT" != 0 ] && [ $((NOW - MT)) -gt 1500 ]; then
      $NOTIFY ALERT "$WORKER STALL? $RUN" "metrics.jsonl not advanced in $(((NOW-MT)/60)) min while pid $TRAINING alive (fp=$FP). Per-run watchdog should SIGTERM shortly; flagging directly."
    fi
    # periodic heartbeat (~every 2h)
    if [ $((TICK % 4)) -eq 0 ]; then
      $NOTIFY INFO "$WORKER heartbeat" "current run=$RUN fp=$FP (last metric $(((NOW-MT)/60)) min ago)"
    fi
  done ) &
HEARTBEAT=$!

# ---------------------- run the manifest ----------------------
$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --worker_name "$WORKER"

kill $SIDECAR $HEARTBEAT 2>/dev/null || true
# final authoritative sync (include encoders — stage 2 warm-starts from them).
# Scoped to owned runs for the same reason as the sidecar: a full-tree push here was the
# second clobber path, firing exactly when a box finished and had the most stale copies.
push_owned

# ---- completion GATE: only self-stop if EVERY run in the manifest is verified ----
# Delegates to launch_v2_wave.py --check_complete so the gate and the skip logic share ONE
# definition of "done". They used to differ: this gate accepted only a LOCAL verified.json,
# while the launcher also accepts an S3 marker, an anchor exemption, and FP>=98% of budget.
# Anything complete via those routes was reported "missing" forever, so the box never
# self-terminated and billed indefinitely while emitting a plausible-looking INCOMPLETE alert.
SUMMARY=$($PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --worker_name "$WORKER" --check_complete)
MISSING=$(echo "$SUMMARY" | $PY -c "import sys,json;print(len(json.load(sys.stdin)['missing']))")
DONE=$(echo "$SUMMARY" | $PY -c "import sys,json;print(len(json.load(sys.stdin)['done']))")
MISSING_LIST=$(echo "$SUMMARY" | $PY -c "import sys,json;print(', '.join(json.load(sys.stdin)['missing']) or 'none')")
DONE_LIST=$(echo "$SUMMARY" | $PY -c "import sys,json;print(', '.join(json.load(sys.stdin)['done']))")

if [ "$MISSING" -eq 0 ]; then
  $NOTIFY DONE "$WORKER manifest COMPLETE ($DONE runs)" "All runs verified complete. Box self-stopping. Done: $DONE_LIST"
  echo "PHASE2 WORKER $WORKER ALL VERIFIED $(date -u)"
  sudo shutdown -h now
else
  $NOTIFY ALERT "$WORKER INCOMPLETE — box STAYING UP" "$DONE verified, $MISSING NOT complete: $MISSING_LIST. Box left running for inspection/resume (NOT shutting down)."
  echo "PHASE2 WORKER $WORKER INCOMPLETE ($MISSING missing) — staying up $(date -u)"
  exit 1
fi

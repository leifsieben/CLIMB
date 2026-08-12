#!/bin/bash
# Experiment B driver (runs ON the box, detached). Self-contained: pretrain the 3 wiki_real runs
# (single-split eval + verified.json via launch_v2_wave), 5-fold CV each, sync ENCODERS to S3
# (durable checkpoints), then — gated on 3/3 verified+CV — stop the box. Never stops on failure.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
MANIFEST=experiments/climb_v2_expB/manifest.json
S3=s3://climb-s3-bucket/experiments/climb_v2_expB
LOG=/home/ec2-user/wiki/expB_wave.log
RUNS="wiki_real_8M wiki_real_8M_s1 wiki_real_8M_s2"
say(){ echo "[expB $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

bash scripts/notify.sh INFO "ExpB wave START" "3 wiki_real runs (8M FP) + 5-fold CV" || true
$PY scripts/build_expB_manifest.py --out "$MANIFEST" >>"$LOG" 2>&1

# Phase 1: pretrain + single-split eval + verified.json + S3 backup
$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" \
    --run_id wiki_real_8M --run_id wiki_real_8M_s1 --run_id wiki_real_8M_s2 \
    --worker_name expB >>"$LOG" 2>&1
say "pretrain wave exited rc=$?"

# Phase 2: 5-fold CV eval + encoder sync
for RID in $RUNS; do
  RD=experiments/climb_v2_expB/$RID
  [ -d "$RD/encoder" ] || { say "$RID: no encoder, skip"; continue; }
  if [ ! -f "$RD/moleculenet_cv/moleculenet_summary.csv" ]; then
    say "CV eval $RID"
    $PY eval_v2.py --encoder "$RD/encoder" --tokenizer "$RD/tokenizer" \
        --output_dir "$RD/moleculenet_cv" --pool mean --standardize zscore --head mlp \
        --max_length 256 --head_seeds 0 1 2 --cv_folds 5 >>"$LOG" 2>&1 \
        && say "$RID CV ok" || { say "$RID CV FAILED"; bash scripts/notify.sh ALERT "ExpB CV FAILED $RID" "see expB_wave.log" || true; }
    aws s3 cp "$RD/moleculenet_cv" "$S3/$RID/moleculenet_cv" --recursive --only-show-errors || true
  fi
  aws s3 cp "$RD/encoder" "$S3/$RID/encoder" --recursive --only-show-errors && say "$RID encoder -> S3" || say "$RID encoder sync FAILED"
done

# Completion gate
DONE=0
for RID in $RUNS; do
  [ -f "experiments/climb_v2_expB/$RID/verified.json" ] && [ -f "experiments/climb_v2_expB/$RID/moleculenet_cv/moleculenet_summary.csv" ] && { DONE=$((DONE+1)); say "  $RID: VERIFIED + CV"; } || say "  $RID: INCOMPLETE"
done
say "ALL DONE: $DONE/3"
echo "EXPB_ALL_DONE done=$DONE/3"
if [ "$DONE" -eq 3 ]; then
  bash scripts/notify.sh DONE "ExpB COMPLETE (3/3) — box stopping" "wiki_real encoders+evals on S3 under climb_v2_expB; box stopping in 3 min" || true
  sleep 180
  sudo shutdown -h now
else
  bash scripts/notify.sh ALERT "ExpB INCOMPLETE ($DONE/3) — box left up" "inspect expB_wave.log; box NOT stopped" || true
fi

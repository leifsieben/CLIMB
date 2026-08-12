#!/bin/bash
# Per-seed Exp B driver (one wiki_real run per box, for parallelizing across instances).
# Pretrain ONE run + 5-fold CV + sync encoder to S3, then self-stop the box (gated on verified+CV).
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
RID=${1:?usage: expB_seed.sh <run_id>}
MANIFEST=experiments/climb_v2_expB/manifest.json
S3=s3://climb-s3-bucket/experiments/climb_v2_expB
LOG=/home/ec2-user/wiki/expB_${RID}.log
say(){ echo "[expB $RID $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

bash scripts/notify.sh INFO "ExpB $RID START" "single-seed on $(hostname)" || true
$PY scripts/build_expB_manifest.py --out "$MANIFEST" >>"$LOG" 2>&1
$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --run_id "$RID" --worker_name "expB_$RID" >>"$LOG" 2>&1
say "pretrain+eval exited rc=$?"

RD=experiments/climb_v2_expB/$RID
if [ -d "$RD/encoder" ] && [ ! -f "$RD/moleculenet_cv/moleculenet_summary.csv" ]; then
  say "CV eval"
  $PY eval_v2.py --encoder "$RD/encoder" --tokenizer "$RD/tokenizer" --output_dir "$RD/moleculenet_cv" \
      --pool mean --standardize zscore --head mlp --max_length 256 --head_seeds 0 1 2 --cv_folds 5 >>"$LOG" 2>&1 \
      && say "CV ok" || say "CV FAILED"
fi
aws s3 cp "$RD/moleculenet_cv" "$S3/$RID/moleculenet_cv" --recursive --only-show-errors || true
aws s3 cp "$RD/encoder" "$S3/$RID/encoder" --recursive --only-show-errors || true

if [ -f "$RD/verified.json" ] && [ -f "$RD/moleculenet_cv/moleculenet_summary.csv" ]; then
  say "COMPLETE — stopping box in 2 min (encoder+CV on S3)"
  bash scripts/notify.sh DONE "ExpB $RID COMPLETE — box stopping" "encoder+CV on S3 under climb_v2_expB/$RID" || true
  echo "EXPB_SEED_DONE $RID"
  sleep 120; sudo shutdown -h now
else
  say "INCOMPLETE — box left up for inspection"
  bash scripts/notify.sh ALERT "ExpB $RID INCOMPLETE — box left up" "inspect $LOG" || true
  echo "EXPB_SEED_INCOMPLETE $RID"
fi

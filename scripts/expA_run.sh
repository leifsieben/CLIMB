#!/bin/bash
# Experiment A driver (runs ON the GPU box, detached). Two phases per the wave's own safeguards:
#   1) launch_v2_wave.py: pretrain each run under a stall+ceiling watchdog, single-split eval,
#      write verified.json ONLY on budget-reached + eval-ok, back up essentials+moleculenet to S3.
#   2) 5-fold scaffold CV eval on each resulting encoder (the paper HEADLINE metric), backed up to
#      moleculenet_cv/ on S3 — matching every existing phase-2 arm.
#
# Idempotent: phase 1 skips verified runs; phase 2 skips runs whose moleculenet_cv already exists.
# Detach with: setsid bash scripts/expA_run.sh > .../expA_run.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
MANIFEST=experiments/climb_v2_expA/manifest.json
S3=s3://climb-s3-bucket/experiments/climb_v2_expA
LOG=/home/ec2-user/synth/expA_wave.log
RUNS="unigram_8M unigram_8M_s1 unigram_8M_s2 corrupt_mlm_8M_s1 corrupt_mlm_8M_s2"

say(){ echo "[expA $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

bash scripts/notify.sh INFO "ExpA wave START" "5 runs (unigram x3 + shuffle s1/s2), 8M FP each; then 5-fold CV" || true

# ---- Phase 1: pretrain + single-split eval + verified.json + S3 backup ----
say "phase 1: launch_v2_wave (pretrain + single-split eval)"
$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --worker_name expA >>"$LOG" 2>&1
say "phase 1 wave exited rc=$?"

# ---- Phase 2: 5-fold scaffold CV eval (headline) per encoder ----
for RID in $RUNS; do
  RD=experiments/climb_v2_expA/$RID
  if [ ! -d "$RD/encoder" ]; then say "$RID: NO ENCODER (pretrain incomplete) - skip CV"; continue; fi
  if [ -f "$RD/moleculenet_cv/moleculenet_summary.csv" ]; then say "$RID: CV already present - skip"; continue; fi
  say "phase 2: CV eval $RID"
  $PY eval_v2.py --encoder "$RD/encoder" --tokenizer "$RD/tokenizer" \
      --output_dir "$RD/moleculenet_cv" --pool mean --standardize zscore --head mlp \
      --max_length 256 --head_seeds 0 1 2 --cv_folds 5 >>"$LOG" 2>&1 \
      && say "$RID CV ok" || { say "$RID CV FAILED"; bash scripts/notify.sh ALERT "ExpA CV FAILED $RID" "see expA_wave.log" || true; }
  aws s3 sync "$RD/moleculenet_cv" "$S3/$RID/moleculenet_cv" --only-show-errors || true
done

# ---- Completion report (achieved-work, not file-exists) ----
say "completion check:"
DONE=0; TOT=0
for RID in $RUNS; do
  TOT=$((TOT+1))
  V=experiments/climb_v2_expA/$RID/verified.json
  CV=experiments/climb_v2_expA/$RID/moleculenet_cv/moleculenet_summary.csv
  if [ -f "$V" ] && [ -f "$CV" ]; then DONE=$((DONE+1)); say "  $RID: VERIFIED + CV"; else say "  $RID: INCOMPLETE (verified=$([ -f "$V" ] && echo y || echo n) cv=$([ -f "$CV" ] && echo y || echo n))"; fi
done
say "ALL DONE: $DONE/$TOT complete"
bash scripts/notify.sh DONE "ExpA wave COMPLETE ($DONE/$TOT)" "encoders on box; single+CV evals on S3 under climb_v2_expA" || true
echo "EXPA_ALL_DONE done=$DONE/$TOT"

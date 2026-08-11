#!/bin/bash
# Re-evaluate the Experiment-A frozen-probe COMPARATORS with the CURRENT native-unit eval_v2, so the
# regression ladder is internally consistent. The phase-2 moleculenet_cv summaries are in NORMALIZED
# units (old eval: QM7 rmse ~0.87), while the new expA arms are native (QM7 rmse ~200) — mixing them
# is invalid. This writes NATIVE re-evals to a SEPARATE path (climb_v2_expA/_baselines/<run>) so the
# paper's existing phase-2 artifacts are left untouched.
#
# Frozen probes only (same protocol as the unigram/shuffle arms): unsup_8M x3, corrupt_mlm_8M (shuffle
# s0), random_baseline x3. e2e_random is end-to-end (different protocol) — handled separately.
# Idempotent: skips a run whose native moleculenet_cv already exists.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
S3P=s3://climb-s3-bucket/experiments/climb_v2_phase2
S3O=s3://climb-s3-bucket/experiments/climb_v2_expA/_baselines
TOK=experiments/climb_v2_expA/unigram_8M/tokenizer   # local tokenizer_10M saved by the expA wave
LOG=/home/ec2-user/synth/expA_baselines.log
RUNS="unsup_8M unsup_8M_s1 unsup_8M_s2 corrupt_mlm_8M random_baseline_00 random_baseline_01 random_baseline_02"

say(){ echo "[base $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -f "$TOK/tokenizer.json" ] || { say "FATAL: tokenizer missing at $TOK"; exit 2; }
bash scripts/notify.sh INFO "ExpA baselines native re-eval START" "7 frozen comparators, 5-fold CV, native units" || true

for RID in $RUNS; do
  OD=experiments/climb_v2_expA/_baselines/$RID
  if [ -f "$OD/moleculenet_cv/moleculenet_summary.csv" ]; then say "$RID: native CV exists, skip"; continue; fi
  mkdir -p "$OD/encoder"
  aws s3 cp "$S3P/$RID/encoder/" "$OD/encoder/" --recursive --only-show-errors
  if [ ! -f "$OD/encoder/model.safetensors" ] && [ ! -f "$OD/encoder/pytorch_model.bin" ]; then say "$RID: NO ENCODER pulled, skip"; continue; fi
  say "native CV eval $RID"
  $PY eval_v2.py --encoder "$OD/encoder" --tokenizer "$TOK" \
      --output_dir "$OD/moleculenet_cv" --pool mean --standardize zscore --head mlp \
      --max_length 256 --head_seeds 0 1 2 --cv_folds 5 >>"$LOG" 2>&1 \
      && say "$RID native CV ok" || { say "$RID native CV FAILED"; bash scripts/notify.sh ALERT "ExpA baseline FAILED $RID" "see expA_baselines.log" || true; }
  aws s3 cp "$OD/moleculenet_cv" "$S3O/$RID/moleculenet_cv" --recursive --only-show-errors
  rm -rf "$OD/encoder"   # free disk; keep only the summary
done
say "BASELINES DONE"
bash scripts/notify.sh DONE "ExpA baselines native re-eval COMPLETE" "native-unit CV for 7 frozen comparators on S3 under climb_v2_expA/_baselines" || true
echo "EXPA_BASELINES_DONE"

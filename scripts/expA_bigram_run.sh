#!/bin/bash
# Experiment A — bigram-resample arm (added 2026-08-11). Closes the ladder gap between
# unigram_resample (marginal only) and shuffle_tokens (full per-molecule multiset): the bigram corpus
# preserves LOCAL token adjacency (a first-order Markov chain fit on the corpus) but destroys the
# per-molecule multiset. Tells us whether local co-occurrence can substitute for the composition.
#
# Waits for (a) the baseline native re-eval to finish (so the two GPU jobs don't contend) and (b) the
# bigram corpus to be materialized, THEN runs 3 bigram runs (8M FP) + 5-fold CV. Idempotent.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
S3=s3://climb-s3-bucket/experiments/climb_v2_expA
LOG=/home/ec2-user/synth/expA_bigram.log
RUNS="bigram_8M bigram_8M_s1 bigram_8M_s2"
say(){ echo "[bigram $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for: baseline re-eval done + bigram corpus materialized"
while true; do
  B=0; M=0
  grep -q EXPA_BASELINES_DONE /home/ec2-user/synth/expA_baselines_run.log 2>/dev/null && B=1
  [ -f /home/ec2-user/synth/bigram_pkl/_diagnostics.json ] && M=1
  [ "$B" = 1 ] && [ "$M" = 1 ] && break
  sleep 60
done
say "prerequisites met (baselines done, bigram corpus ready)"
bash scripts/notify.sh INFO "ExpA bigram wave START" "3 bigram runs (8M FP) + 5-fold CV" || true

$PY scripts/build_expA_manifest.py --out experiments/climb_v2_expA/manifest.json >>"$LOG" 2>&1
say "manifest rebuilt (includes bigram runs)"

$PY scripts/launch_v2_wave.py --manifest experiments/climb_v2_expA/manifest.json \
    --run_id bigram_8M --run_id bigram_8M_s1 --run_id bigram_8M_s2 \
    --worker_name expA_bigram >>"$LOG" 2>&1
say "bigram pretrain wave exited rc=$?"

for RID in $RUNS; do
  RD=experiments/climb_v2_expA/$RID
  [ -d "$RD/encoder" ] || { say "$RID: no encoder (pretrain incomplete), skip CV"; continue; }
  [ -f "$RD/moleculenet_cv/moleculenet_summary.csv" ] && { say "$RID: CV exists, skip"; continue; }
  say "CV eval $RID"
  $PY eval_v2.py --encoder "$RD/encoder" --tokenizer "$RD/tokenizer" \
      --output_dir "$RD/moleculenet_cv" --pool mean --standardize zscore --head mlp \
      --max_length 256 --head_seeds 0 1 2 --cv_folds 5 >>"$LOG" 2>&1 \
      && say "$RID CV ok" || { say "$RID CV FAILED"; bash scripts/notify.sh ALERT "ExpA bigram CV FAILED $RID" "see expA_bigram.log" || true; }
  aws s3 cp "$RD/moleculenet_cv" "$S3/$RID/moleculenet_cv" --recursive --only-show-errors || true
done

DONE=0
for RID in $RUNS; do
  [ -f "experiments/climb_v2_expA/$RID/verified.json" ] && [ -f "experiments/climb_v2_expA/$RID/moleculenet_cv/moleculenet_summary.csv" ] && { DONE=$((DONE+1)); say "  $RID: VERIFIED + CV"; } || say "  $RID: INCOMPLETE"
done
say "ALL DONE: $DONE/3"
bash scripts/notify.sh DONE "ExpA bigram wave COMPLETE ($DONE/3)" "bigram arm on S3 under climb_v2_expA" || true
echo "EXPA_BIGRAM_DONE done=$DONE/3"

#!/usr/bin/env bash
# The 8 missing CBS pretraining-seed replicates.
#
# WHY: sup_minimol, sup_mixed, u2s_minimol and u2s_mixed have only their base CBS dir, so their
# CBS points rest on ONE pretraining seed while every other arm's rest on three. These are the
# _s1/_s2 runs that close that gap (all 8 encoders verified present on S3).
#
# WHY CPU: there is no GPU capacity in us-east-1 right now, on-demand or spot (20 type x AZ
# combinations tried). The CBS frozen probe is a featurization pass over ~5k molecules plus an
# MLP head, so it runs fine on CPU -- eval_v2.py picks its device via torch.cuda.is_available().
# Waiting for a GPU would cost more wall-clock than just running it here.
#
# SHARD/NSHARD split the list across boxes. Gated: shuts down only when every assigned run has a
# real cbs_nef1_MEAN, never on merely reaching the end.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/cbs_seeds.log
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
say() { echo "[cbs] $(date -u +%FT%TZ) $*" >> "$LOG"; }

ALL=(skip_minimol_full_8M_s1 skip_minimol_full_8M_s2 skip_mixed_8M_s1 skip_mixed_8M_s2
     u2s_minimol_full_from8M_s1 u2s_minimol_full_from8M_s2 u2s_mixed_from8M_s1 u2s_mixed_from8M_s2)
RUNS=(); i=0
for r in "${ALL[@]}"; do [ $((i % NSHARD)) -eq "$SHARD" ] && RUNS+=("$r"); i=$((i+1)); done
say "start shard $SHARD/$NSHARD: ${RUNS[*]}"

[ -f data/cbs.csv ] || aws s3 cp s3://climb-s3-bucket/datasets/cbs.csv data/cbs.csv --only-show-errors
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ -f data/cbs.csv ] || { say "FATAL no cbs.csv -> staying UP"; exit 1; }

done_ok() {  # real metric, not file existence
  ~/venvs/climb/bin/python -c "
import json,sys
try: sys.exit(0 if json.load(open('figure_data/cbs_benchmark/$1/moleculenet_cv/suite_summary.json')).get('cbs_nef1_MEAN') is not None else 1)
except Exception: sys.exit(1)" 2>/dev/null
}

ok=0
for r in "${RUNS[@]}"; do
  if done_ok "$r"; then say "SKIP $r (already done)"; ok=$((ok+1)); continue; fi
  ENC=figure_data/_stage_cbs/$r/encoder
  if [ ! -f "$ENC/model.safetensors" ]; then
    mkdir -p "$ENC"
    aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2/$r/encoder "$ENC" --only-show-errors
  fi
  [ -f "$ENC/model.safetensors" ] || { say "ERROR $r: no encoder after sync"; continue; }
  say "CBS frozen: $r"
  ~/venvs/climb/bin/python eval_v2.py --encoder "$ENC" --tokenizer figure_data/_tokenizer \
    --output_dir "figure_data/cbs_benchmark/$r/moleculenet_cv" --head mlp --head_seeds 0 1 2 \
    --task_csv data/cbs.csv --task_name cbs --task_type classification \
    --cv_folds 5 --cv_scheme provided >> "$LOG" 2>&1
  if done_ok "$r"; then
    echo "{\"run\":\"$r\",\"metric\":\"nef1\",\"cv\":\"provided-5fold\",\"panel\":\"cbs_seed_replicates\"}" \
      > "figure_data/cbs_benchmark/$r/verified.json"
    aws s3 cp --recursive "figure_data/cbs_benchmark/$r" \
      "s3://climb-s3-bucket/experiments/cbs_benchmark/$r" --only-show-errors
    say "OK $r"; ok=$((ok+1))
  else
    say "FAIL $r"
  fi
  rm -rf "figure_data/_stage_cbs/$r"
done

say "DONE $ok/${#RUNS[@]}"
if [ "$ok" -eq "${#RUNS[@]}" ]; then say "all verified -> shutdown"; sudo shutdown -h now
else say "incomplete -> staying UP for inspection"; fi

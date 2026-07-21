#!/usr/bin/env bash
# E1 (eval ceiling) at publication strength, on a GPU box.
#
# The laptop run used ONE finetune seed on three small datasets (BACE 1.5k, BBBP 2k, ESOL 1.1k)
# while measuring spreads of 0.008-0.077 -- enough to say "no evidence the frozen probe hid a
# scaling effect", not enough to assert it. This adds:
#   * 3 finetune seeds, so the finetuned point carries a real error bar;
#   * HIV (41k molecules, ~20x BBBP), by far the best-resolved task available.
# Finetuning unfreezes a 41M-parameter encoder, so HIV is impractical on CPU -- hence GPU.
#
# Encoders/frozen baselines are pulled from S3 because this box's local tree belongs to whatever
# wave it last ran, not to climb_v2_phase2.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
ROOT=experiments/_e1_ceiling_src
OUT=experiments/_e1_ceiling
RUNS="random_baseline_00 unsup_2M unsup_8M unsup_24M"

mkdir -p "$ROOT" "$OUT"
aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$ROOT/_tokenizer" --only-show-errors
for r in $RUNS; do
  aws s3 sync "$S3/$r/encoder" "$ROOT/$r/encoder" --only-show-errors
  # frozen-probe baseline for the comparison line
  aws s3 sync "$S3/$r/moleculenet" "$ROOT/$r/moleculenet" \
      --exclude "*" --include "moleculenet_summary.csv" --only-show-errors
  echo "  staged $r ($(ls "$ROOT/$r/encoder" 2>/dev/null | wc -l) encoder files)"
done

$PY scripts/run_eval_ceiling.py \
  --results_root "$ROOT" \
  --run_ids $RUNS \
  --tasks BBBP BACE ESOL HIV \
  --seeds 0 1 2 \
  --tokenizer "$ROOT/_tokenizer" \
  --output_dir "$OUT"
rc=$?

aws s3 sync "$OUT" s3://climb-s3-bucket/derived/eval_ceiling --only-show-errors
bash scripts/notify.sh "$([ $rc -eq 0 ] && echo DONE || echo ALERT)" \
  "E1 eval-ceiling (3 seeds, incl HIV) rc=$rc" \
  "Results at s3://climb-s3-bucket/derived/eval_ceiling/eval_ceiling.csv"
echo "E1_GPU_DONE rc=$rc"
# Propagate the real status: the script previously ended on `echo`, so its exit code was the
# echo's (0) and a caller checking $? saw success even when the run had failed.
exit "$rc"

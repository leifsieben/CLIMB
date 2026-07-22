#!/usr/bin/env bash
# E1, second half: the SUPERVISED ladder through the same eval-ceiling harness.
#
# Why this exists. The first E1 run covered only random_baseline + the unsup_only ladder, so the
# figure has exactly one frozen line and one fine-tuned line per task. That answers a
# ladder-SHAPE question ("does fine-tuning reveal a slope the frozen probe was hiding?") but it
# cannot answer H5, which is a regime-ORDERING question: "is `sup_only <= unsup_only` a
# frozen-probe artifact?". Testing an ordering flip needs both regimes in both conditions --
# four lines per task -- because a single pair has nothing to flip against.
#
# Recipe: `dense`, the best sup_only recipe by mean lift over no_pretrain in A1. Matching the
# unsup ladder budget for budget (2M / 8M / 24M) is what makes the frozen-vs-finetuned gaps
# comparable between regimes.
#
# Writes to a SEPARATE output dir and the notebook merges the two CSVs, rather than re-running
# the 48 unsup fine-tunes that already exist. Only the `finetune` column is taken from here --
# frozen values are read per head seed from the run summaries.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
ROOT=experiments/_e1_ceiling_src
OUT=experiments/_e1_ceiling_sup
RECIPE=${RECIPE:-dense}
RUNS="skip_${RECIPE}_2M skip_${RECIPE}_8M skip_${RECIPE}_24M"

mkdir -p "$ROOT" "$OUT"
aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$ROOT/_tokenizer" --only-show-errors
for r in $RUNS; do
  aws s3 sync "$S3/$r/encoder" "$ROOT/$r/encoder" --only-show-errors
  aws s3 sync "$S3/$r/moleculenet" "$ROOT/$r/moleculenet" \
      --exclude "*" --include "moleculenet_summary.csv" --only-show-errors
  n=$(ls "$ROOT/$r/encoder" 2>/dev/null | wc -l)
  echo "  staged $r ($n encoder files)"
  # A missing encoder would silently produce finetune=None for every seed and a figure with a
  # gap where the comparison should be. Fail loudly instead.
  if [ ! -f "$ROOT/$r/encoder/model.safetensors" ]; then
    echo "FATAL: $r has no encoder weights in S3"; exit 2
  fi
done

$PY scripts/run_eval_ceiling.py \
  --results_root "$ROOT" \
  --run_ids $RUNS \
  --tasks BBBP BACE ESOL HIV \
  --seeds 0 1 2 \
  --tokenizer "$ROOT/_tokenizer" \
  --output_dir "$OUT"
rc=$?

aws s3 sync "$OUT" s3://climb-s3-bucket/derived/eval_ceiling_sup --only-show-errors
bash scripts/notify.sh "$([ $rc -eq 0 ] && echo DONE || echo ALERT)" \
  "E1 sup_only ladder (recipe=$RECIPE, 3 seeds, incl HIV) rc=$rc" \
  "Results at s3://climb-s3-bucket/derived/eval_ceiling_sup/eval_ceiling.csv"
echo "E1_SUP_DONE rc=$rc"
exit "$rc"

#!/usr/bin/env bash
# Re-run evaluation for every run in a results dir that has a saved encoder (or the
# ECFP4 anchor), WITHOUT re-pretraining. Recovers from an eval-only failure (e.g. the
# ModernBERT torch.compile/gcc issue). Skips runs whose eval CSV already exists.
#
# Usage (on the instance):  bash scripts/reeval_v2.sh [results_root] [s3_backup_root]
set -uo pipefail

RESULTS_ROOT="${1:-experiments/climb_v2}"
S3_ROOT="${2:-s3://climb-s3-bucket/experiments/climb_v2}"
PY="${PY:-/home/ec2-user/venvs/climb/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1   # hard-disable torch.compile/inductor (broken toolchain)

for d in "$RESULTS_ROOT"/*/; do
  rid=$(basename "$d")
  eval_dir="${d}moleculenet"
  if [ -f "$eval_dir/moleculenet_summary.csv" ]; then echo "SKIP $rid (eval exists)"; continue; fi
  if [ "$rid" = "ecfp4_anchor" ]; then
    echo "=== EVAL $rid (ecfp4+xgb) ==="
    $PY eval_v2.py --output_dir "$eval_dir" --featurizer ecfp4 --head xgb --head_seeds 0 1 2 || { echo "FAILED $rid"; continue; }
  elif [ -f "${d}encoder/model.safetensors" ] && [ -d "${d}tokenizer" ]; then
    echo "=== EVAL $rid (encoder) ==="
    $PY eval_v2.py --encoder "${d}encoder" --tokenizer "${d}tokenizer" --output_dir "$eval_dir" \
        --pool mean --standardize zscore --head mlp --max_length 256 --head_seeds 0 1 2 || { echo "FAILED $rid"; continue; }
  else
    echo "SKIP $rid (no encoder/tokenizer)"; continue
  fi
  aws s3 sync "$eval_dir" "$S3_ROOT/$rid/moleculenet" >/dev/null 2>&1 && echo "  backed up $rid"
done
echo "=== re-eval complete ==="

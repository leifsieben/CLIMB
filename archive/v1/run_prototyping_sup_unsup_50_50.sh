#!/usr/bin/env bash
set -euo pipefail

# Helper script that runs a 50/50 mixture of supervised and unsupervised pretraining
# using the already-tokenized pickles in `local_prototyping_data`.

ROOT="$(cd "$(dirname "$0")" && pwd)"
TOKENIZER="${ROOT}/local_prototyping_data/tokenizer"
UNSUP="${ROOT}/local_prototyping_data/unsupervised_tokenized.pkl"
SUP="${ROOT}/local_prototyping_data/supervised_tokenized.pkl"
CONFIG="${ROOT}/configs/config_prototyping_50_50.yaml"
OUTPUT="${ROOT}/experiments/prototyping_sup_unsup_50_50"

mkdir -p "$OUTPUT"

exec python3 train_model.py \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --unsup_data "$UNSUP" \
  --sup_data "$SUP" \
  --unsup_weight 0.5 \
  --sup_weight 0.5 \
  --output "$OUTPUT" \
  --task mlm \
  --log_file "$OUTPUT/train.log"

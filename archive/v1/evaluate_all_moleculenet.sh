#!/usr/bin/env bash
set -euo pipefail

# Evaluate a pretrained model on all MoleculeNet datasets.
# Usage: ./evaluate_all_moleculenet.sh /path/to/pretrained_model /path/to/output_root [tokenizer_path]
# Example:
#   ./evaluate_all_moleculenet.sh \
#     experiments/prototyping_sup_unsup_50_50 \
#     experiments/moleculenet_evals \
#     tokenizer
#
# Requirements:
#   - deepchem, torch, transformers installed
#   - evaluate_model.py present in this repo

MODEL_DIR="${1:-}"
OUTPUT_ROOT="${2:-}"
TOKENIZER_PATH="${3:-$MODEL_DIR}"

if [[ -z "$MODEL_DIR" || -z "$OUTPUT_ROOT" ]]; then
  echo "Usage: $0 <pretrained_model_dir> <output_root_dir> [tokenizer_path]" >&2
  exit 1
fi

DATASETS=(
  QM7
  QM8
  QM9
  Tox21
  BBBP
  ToxCast
  SIDER
  ClinTox
  HIV
  BACE
  MUV
  PCBA
  ESOL
  FreeSolv
  Lipophilicity
)

mkdir -p "$OUTPUT_ROOT"

for ds in "${DATASETS[@]}"; do
  echo "=== Evaluating on $ds ==="
  python3 evaluate_model.py \
    --pretrained_model "$MODEL_DIR" \
    --dataset moleculenet \
    --dataset_name "$ds" \
    --output "$OUTPUT_ROOT/$ds" \
    --tokenizer "$TOKENIZER_PATH" \
    --freeze_encoder
done

echo "All evaluations complete. Aggregate results: see each dataset folder and any all_evaluations.txt files under $OUTPUT_ROOT."

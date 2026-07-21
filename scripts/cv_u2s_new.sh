#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv_sanity/bin/python
TOK=figure_data/_tokenizer
CORE="ESOL BBBP BACE Tox21 QM7"
for run in u2s_dense_from2M u2s_dense_from8M u2s_dense_from24M u2s_dense_plus_sparse_from24M; do
  enc=figure_data/climb_v2_phase2/$run/encoder
  [ -f "$enc/model.safetensors" ] || { echo "SKIP $run (no enc)"; continue; }
  echo "=== CV $run ==="
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" \
    --output_dir figure_data/climb_v2_phase2/$run/moleculenet_cv \
    --cv_folds 5 --head_seeds 0 1 2 --pool mean --standardize zscore --head mlp --max_length 256 \
    --datasets $CORE 2>&1 | grep -iE "wrote|error|fold|FAIL" | tail -8
done
echo "CV_U2S_DONE"

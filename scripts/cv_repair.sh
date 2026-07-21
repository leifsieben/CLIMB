#!/bin/bash
# Re-run 5-fold scaffold CV for the two runs whose CV is missing or was computed on the wrong
# encoder, using the same protocol as cv_eval_local.py (7-task CORE, 5 folds, 3 head seeds).
#
#   skip_mixed_24M  — its local encoder was downloaded at 05:03 while the run did not finish
#                     until 11:39, so the existing CV scored a mid-training checkpoint. The two
#                     encoders differ by md5, so this is a real invalidation, not a timestamp
#                     artefact.
#   skip_dense_48M  — verified complete at 10:20; never downloaded and never evaluated.
cd "$(dirname "$0")/.."
PY=.venv_sanity/bin/python
TOK=figure_data/_tokenizer
CORE="ESOL Lipophilicity QM7 BBBP BACE Tox21 HIV"

for run in skip_mixed_24M skip_dense_48M; do
  enc=figure_data/climb_v2_phase2/$run/encoder
  [ -f "$enc/model.safetensors" ] || { echo "SKIP $run (no encoder weights)"; continue; }
  # DeepChem featurization caches collide across concurrent/repeat runs
  rm -rf "${TMPDIR:-/tmp}"/*-featurized 2>/dev/null
  echo "=== CV $run ==="
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" \
    --output_dir figure_data/climb_v2_phase2/$run/moleculenet_cv \
    --cv_folds 5 --head_seeds 0 1 2 --pool mean --standardize zscore --head mlp --max_length 256 \
    --datasets $CORE 2>&1 | grep -iE "wrote|error|FAIL" | tail -6
done
echo "CV_REPAIR_DONE"

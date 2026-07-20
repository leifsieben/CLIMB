#!/bin/bash
cd /Users/lsieben/VSCode/CLIMB
PY=.venv_sanity/bin/python; TOK=figure_data/_tokenizer
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
TASKS="ESOL Lipophilicity QM7 BBBP BACE Tox21"     # A2's core tasks; HIV folded in with the final pass (41k = slow)
RUNS="unsup_2M unsup_24M skip_dense_2M skip_dense_plus_sparse_2M skip_dense_plus_sparse_24M \
skip_minimol_full_2M skip_minimol_full_24M skip_mixed_2M skip_mixed_48M \
skip_sparse_all_2M skip_sparse_all_24M skip_sparse_all_48M"
for run in $RUNS; do
  enc=figure_data/climb_v2_phase2/$run/encoder
  [ -f "$enc/model.safetensors" ] || { echo "== download $run =="; aws s3 sync $S3/$run/encoder $enc --quiet; }
  [ -f "$enc/model.safetensors" ] || { echo "SKIP $run (no encoder in S3)"; continue; }
  echo "===== CV $run ====="
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" \
    --output_dir figure_data/climb_v2_phase2/$run/moleculenet_cv \
    --cv_folds 5 --head_seeds 0 1 2 --pool mean --standardize zscore --head mlp --max_length 256 \
    --datasets $TASKS 2>&1 | grep -iE "= [0-9]|wrote|error|FAIL" | tail -8
done
echo CV_ALL_BUDGETS_DONE

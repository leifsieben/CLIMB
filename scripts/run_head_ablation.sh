#!/bin/bash
# Head ablation: FFN vs XGBoost on FROZEN CLIMB embeddings.
#
# The whole notebook uses an MLP (FFN) downstream head. This asks whether XGBoost on the same
# frozen embeddings changes the story -- and whether it lets CLIMB clear the Morgan+XGBoost anchors
# it otherwise loses to. It is the SAME eval_v2 harness, 5-fold scaffold CV, pool=mean, zscore
# standardize as every other run; ONLY --head changes (mlp -> xgb). The Morgan anchors
# (ecfp4_anchor, fp_desc_anchor) are already XGBoost and already CV-scored in figure_data.
#
# The FFN CLIMB numbers used by the notebook table come from the production runs in
# figure_data/climb_v2_phase2/{unsup_8M,skip_dense_plus_sparse_8M}/moleculenet_cv (== the A1.b
# numbers). This script (re)generates only the XGBoost-on-embeddings side, into the tracked
# analysis/head_ablation/ dir so it travels with the repo. The encoder runs on MPS locally
# (eval_v2 falls back cuda->mps->cpu), which differs from the production CUDA embeddings by <=0.01.
cd /Users/lsieben/VSCode/CLIMB
PY=.venv_sanity/bin/python
TOK=figure_data/_tokenizer
TASKS="ESOL BBBP BACE Tox21 QM7 HIV"
for run in unsup_8M skip_dense_plus_sparse_8M; do
  enc=figure_data/climb_v2_phase2/$run/encoder
  [ -f "$enc/model.safetensors" ] || { echo "MISSING encoder for $run"; continue; }
  echo "===== XGBoost-on-embeddings: $run ====="
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" \
    --output_dir analysis/head_ablation/${run}_xgb/moleculenet_cv \
    --cv_folds 5 --head_seeds 0 1 2 --pool mean --standardize zscore --head xgb \
    --max_length 256 --datasets $TASKS
done
echo "HEAD_ABLATION_DONE"

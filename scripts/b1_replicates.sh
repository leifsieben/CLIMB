#!/bin/bash
# B1 label-efficiency REPLICATES using phase2 8M encoders (consistent with A1/A2).
# For each regime x train-size x subsample_seed, run the frozen-probe eval on a fresh
# random draw of the train subset -> error bars = spread across draws (+3 head seeds each).
# Full train (n=0): subsampling is a no-op, so 1 seed suffices. Writes to labeleff_rep/.
cd /Users/lsieben/VSCode/CLIMB
PY=.venv_sanity/bin/python
TOK=figure_data/_tokenizer
TASKS="ESOL BBBP BACE Tox21 QM7"
OUT=figure_data/climb_v2_labeleff_rep
mkdir -p "$OUT"
# regime:encoder_run  (random/unsup/sup use phase2 8M encoders; ecfp4 = classical, no encoder)
run_enc() {  # $1=name $2=encoder_run $3=n $4=seed
  local name=$1 run=$2 n=$3 seed=$4
  local enc=figure_data/climb_v2_phase2/$run/encoder
  local out=$OUT/${name}_n${n}_s${seed}/moleculenet
  local sub=""; [ "$n" != "0" ] && sub="--train_subsample $n"
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" --output_dir "$out" \
    $sub --subsample_seed $seed --head_seeds 0 1 2 --pool mean --standardize zscore --head mlp \
    --max_length 256 --datasets $TASKS 2>&1 | grep -iE "= [0-9]|wrote|error" | tail -3
}
run_ecfp() {  # $1=n $2=seed
  local n=$1 seed=$2
  local out=$OUT/ecfp4_n${n}_s${seed}/moleculenet
  local sub=""; [ "$n" != "0" ] && sub="--train_subsample $n"
  $PY eval_v2.py --output_dir "$out" --featurizer ecfp4 --head xgb \
    $sub --subsample_seed $seed --head_seeds 0 1 2 --datasets $TASKS 2>&1 | grep -iE "= [0-9]|wrote|error" | tail -3
}
for n in 100 300 1000 3000 0; do
  # subsampled sizes get 3 draws; full (n=0) gets 1 (train identical across draws)
  SEEDS="0 1 2"; [ "$n" = "0" ] && SEEDS="0"
  for seed in $SEEDS; do
    echo "===== n=$n seed=$seed ====="
    run_enc random   random_baseline_00 "$n" "$seed"
    run_enc unsup    unsup_8M           "$n" "$seed"
    run_enc sup      skip_dense_8M      "$n" "$seed"
    run_ecfp "$n" "$seed"
  done
done
echo B1_REPLICATES_DONE

#!/bin/bash
# ⚠️ DEPRECATED / DO NOT RE-RUN — kept for provenance only.
# This writes the ABSOLUTE-budget {100,300,1000,3000,full} label-efficiency data (climb_v2_labeleff_v2),
# which CAPS on small tasks (duplicate points). Superseded by scripts/label_eff_fractions.py (per-task
# fractions 5/10/25/50/100%). See README §E12 / Fig B1p1. Re-running this re-introduces the capped data.
#
# B1 / B1.1 label-efficiency, v2 — PRIMARY regimes, 7 tasks, train AND test metrics.
#
# Changes from b1_replicates.sh:
#   * adds unsup2sup (u2s_dense_from8M), the realistic recipe, which the original sweep never
#     covered -- it is one of the five primary models and was silently absent from B1;
#   * drops the Morgan/ecfp4 anchor (A1 and T1 already carry the classical comparator);
#   * 7 tasks, so B1 has HIV and Lipophilicity like every other panel;
#   * train metrics come for free now that eval_v2 emits "<metric>_train" -- that is what makes
#     B1.1 possible. Test alone cannot separate "the head cannot FIT this many labels" from
#     "it fits and fails to GENERALISE", which is the entire question at small N.
#
# Encoders are the 8M rung, matching A1/A2. sup uses skip_dense_8M and unsup2sup uses
# u2s_dense_from8M so the two SFT arms share a recipe and differ only in the MLM stage.
#
# NOTE: no_pretrain_e2e (the 5th primary) is deliberately NOT here -- it needs a fine-tuned
# sweep, not this frozen-probe path.
cd /home/ec2-user/CLIMB 2>/dev/null || cd /Users/lsieben/VSCode/CLIMB
PY=${PY:-.venv_sanity/bin/python}
TOK=${TOK:-figure_data/_tokenizer}
SRC=${SRC:-figure_data/climb_v2_phase2}
OUT=${OUT:-figure_data/climb_v2_labeleff_v2}
TASKS=${TASKS:-"ESOL Lipophilicity QM7 BBBP BACE Tox21 HIV"}
mkdir -p "$OUT"

run_enc() {  # name encoder_run n seed
  local name=$1 run=$2 n=$3 seed=$4
  local enc=$SRC/$run/encoder
  local out=$OUT/${name}_n${n}_s${seed}/moleculenet
  [ -f "$out/suite_summary.json" ] && { echo "  skip ${name}_n${n}_s${seed} (done)"; return; }
  [ -f "$enc/model.safetensors" ] || { echo "  MISSING encoder for $name ($run)"; return; }
  local sub=""; [ "$n" != "0" ] && sub="--train_subsample $n"
  $PY eval_v2.py --encoder "$enc" --tokenizer "$TOK" --output_dir "$out" \
    $sub --subsample_seed "$seed" --head_seeds 0 1 2 --pool mean --standardize zscore \
    --head mlp --max_length 256 --datasets $TASKS 2>&1 | grep -iE "= [0-9]|wrote|error" | tail -2
  echo "  done ${name}_n${n}_s${seed}"
}

for n in 100 300 1000 3000 0; do
  SEEDS="0 1 2"; [ "$n" = "0" ] && SEEDS="0"   # full train: subsampling is a no-op
  for seed in $SEEDS; do
    echo "===== n=$n seed=$seed ====="
    run_enc random    random_baseline_00  "$n" "$seed"
    run_enc unsup     unsup_8M            "$n" "$seed"
    run_enc sup       skip_dense_8M       "$n" "$seed"
    run_enc unsup2sup u2s_dense_from8M    "$n" "$seed"
  done
done
echo B1_V2_DONE

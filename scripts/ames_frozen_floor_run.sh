#!/usr/bin/env bash
# The frozen random-encoder floor on Ames/Polaris is 1 seed while every other arm carries 3.
# random_baseline_01 and _02 have no Polaris coverage at all, so fig_C2/fig_D's frozen floor and
# the Ames panel's control rest on a single pretraining seed. Frozen probe, full 28-task track,
# same driver and eval seeds as every other arm.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/ames_frozen_floor.log
say() { echo "[ffloor] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
mkdir -p chemeleon_suite/data/polaris
aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
n=$(grep -c . chemeleon_suite/tasks/polaris_tasks.txt 2>/dev/null || echo 0)
[ "$n" -ge 28 ] || { say "FATAL polaris_tasks.txt has $n tasks -> staying UP"; exit 1; }
build_polaris_venv() {
  for a in 1 2 3; do
    [ -x .venv_polaris/bin/python ] && .venv_polaris/bin/python -c "import numpy,polaris" 2>/dev/null && return 0
    rm -rf .venv_polaris; python3.12 -m venv .venv_polaris
    .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
    .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  done
  .venv_polaris/bin/python -c "import numpy,polaris" 2>/dev/null
}
build_polaris_venv || { say "FATAL polaris venv -> staying UP"; exit 1; }
done_ok() { local f=figure_data/chemeleon_suite/polaris/$1/polaris_scores.csv
  [ -s "$f" ] && [ "$(tail -n +2 "$f" | cut -d, -f1 | sort -u | wc -l)" -ge 20 ]; }
ok=0; total=0
for r in random_baseline_00 random_baseline_01 random_baseline_02; do
  total=$((total+1))
  if done_ok "$r"; then say "SKIP $r"; ok=$((ok+1)); continue; fi
  ENC=figure_data/_stage_ffloor/$r/encoder
  if [ ! -f "$ENC/model.safetensors" ]; then mkdir -p "$ENC"
    for w in climb_v2_phase2 climb_v2_ablation_dedup; do
      aws s3 sync "s3://climb-s3-bucket/experiments/$w/$r/encoder" "$ENC" --only-show-errors
      [ -f "$ENC/model.safetensors" ] && break
    done; fi
  [ -f "$ENC/model.safetensors" ] || { say "ERROR $r: no encoder"; continue; }
  say "frozen Polaris: $r"
  ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track polaris --featurizer encoder \
    --model "$r" --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp \
    --seeds 42 117 709 >> "$LOG" 2>&1
  .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py \
    "figure_data/chemeleon_suite/polaris/$r" >> "$LOG" 2>&1
  if done_ok "$r"; then
    aws s3 cp --recursive "figure_data/chemeleon_suite/polaris/$r" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/$r" --only-show-errors
    say "OK $r"; ok=$((ok+1))
  else say "FAIL $r"; fi
  rm -rf "figure_data/_stage_ffloor/$r"
done
say "DONE $ok/$total"
[ "$ok" -eq "$total" ] && { say "verified -> shutdown"; sudo shutdown -h now; } || say "incomplete -> staying UP"

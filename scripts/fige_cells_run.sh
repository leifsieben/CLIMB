#!/usr/bin/env bash
# fig_E's last 1-seed cells. corrupt_mtr_8M_s1/_s2 have MoleculeACE + CBS but no
# climb_v2_phase2/<run>/moleculenet_cv/, so their BACE/Tox21/QM7 cells draw without a whisker while
# every other cell in that figure is 3-seed; and corrupt_mtr_8M_s2 has no tdcommons/ames row, so
# Ames is 2-seed. Three small runs make fig_E uniformly 3-seed.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/fige_cells.log
say() { echo "[fige] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
mkdir -p chemeleon_suite/data/polaris
[ -f chemeleon_suite/data/polaris/tdcommons__ames.csv ] || \
  aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors

stage() {
  local enc=figure_data/_stage_fige/$1/encoder
  [ -f "$enc/model.safetensors" ] || { mkdir -p "$enc"
    aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$1/encoder" "$enc" --only-show-errors; }
  [ -f "$enc/model.safetensors" ] && echo "$enc"
}
molnet_ok() { [ -s "figure_data/climb_v2_phase2/$1/moleculenet_cv/moleculenet_summary.csv" ]; }
ames_ok() { local f=figure_data/chemeleon_suite/polaris/$1/polaris_scores.csv
  [ -s "$f" ] && tail -n +2 "$f" | cut -d, -f1 | grep -qx "tdcommons/ames"; }

ok=0; total=3
for r in corrupt_mtr_8M_s1 corrupt_mtr_8M_s2; do
  if molnet_ok "$r"; then say "SKIP $r molnet"; ok=$((ok+1)); continue; fi
  ENC=$(stage "$r"); [ -z "$ENC" ] && { say "ERROR $r no encoder"; continue; }
  say "MolNet CV: $r"
  ~/venvs/climb/bin/python eval_v2.py --encoder "$ENC" --tokenizer figure_data/_tokenizer \
    --output_dir "figure_data/climb_v2_phase2/$r/moleculenet_cv" --head mlp --head_seeds 0 1 2 \
    --cv_folds 5 --cv_scheme scaffold >> "$LOG" 2>&1
  if molnet_ok "$r"; then
    aws s3 cp --recursive "figure_data/climb_v2_phase2/$r/moleculenet_cv" \
      "s3://climb-s3-bucket/experiments/climb_v2_phase2/$r/moleculenet_cv" --only-show-errors
    say "OK $r molnet"; ok=$((ok+1))
  else say "FAIL $r molnet"; fi
done

r=corrupt_mtr_8M_s2
if ames_ok "$r"; then say "SKIP $r ames"; ok=$((ok+1)); else
  ENC=$(stage "$r")
  if [ -n "$ENC" ]; then
    say "Polaris (Ames): $r"
    ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track polaris --featurizer encoder \
      --model "$r" --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp \
      --seeds 42 117 709 >> "$LOG" 2>&1
    if [ -x .venv_polaris/bin/python ]; then
      .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py \
        "figure_data/chemeleon_suite/polaris/$r" >> "$LOG" 2>&1
    fi
    if ames_ok "$r"; then
      aws s3 cp --recursive "figure_data/chemeleon_suite/polaris/$r" \
        "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/$r" --only-show-errors
      say "OK $r ames"; ok=$((ok+1))
    else say "FAIL $r ames"; fi
  fi
fi
say "DONE $ok/$total"
[ "$ok" -eq "$total" ] && { say "verified -> shutdown"; sudo shutdown -h now; } || say "incomplete -> staying UP"

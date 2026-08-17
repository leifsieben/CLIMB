#!/usr/bin/env bash
# Complete the catastrophic-forgetting arm (s2u_dense_from8M_s{0,1,2}) on the canonical panels.
#
# Two fixes over the first pass:
#  1. MolNet as 5-fold scaffold CV into moleculenet_cv/ (the first pass only produced the single
#     hold-out under moleculenet/, which is NOT comparable to any other arm and must not be pooled).
#     Only the three MolNet panels the six-panel suite uses: BACE, Tox21, QM7.
#  2. MoleculeACE, which failed the first time purely because chemeleon_suite/tasks/moleculeace_tasks.txt
#     was never staged (the code-only deploy ships no .txt; the 30 CSVs were present).
# CBS already landed on the correct protocol and is left alone.
# Gated self-shutdown: only when all 3 seeds have BOTH moleculenet_cv and MoleculeACE.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/s2u_complete.log
echo "[s2u-complete] start $(date -u +%FT%TZ)" >> "$LOG"

[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer; aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

for s in s0 s1 s2; do
  run=s2u_dense_from8M_$s
  enc=experiments/climb_v2_phase2/$run/encoder
  if [ ! -f "$enc/model.safetensors" ]; then echo "[s2u-complete] MISSING $enc" >> "$LOG"; continue; fi

  # ---- 1. MolNet 5-fold scaffold CV (same protocol as every other arm) ----
  out=figure_data/climb_v2_phase2/$run/moleculenet_cv
  if [ ! -f "$out/moleculenet_summary.csv" ]; then
    echo "[s2u-complete] molnet CV: $run" >> "$LOG"
    ~/venvs/climb/bin/python eval_v2.py --encoder "$enc" --tokenizer figure_data/_tokenizer \
      --output_dir "$out" --head mlp --head_seeds 0 1 2 \
      --datasets BACE Tox21 QM7 --cv_folds 5 >> "$LOG" 2>&1
    aws s3 cp --recursive "$out" "s3://climb-s3-bucket/experiments/climb_v2_phase2/$run/moleculenet_cv" --only-show-errors
  fi

  # ---- 2. MoleculeACE frozen probe ----
  if [ ! -f "figure_data/chemeleon_suite/moleculeace/$run/verified.json" ]; then
    echo "[s2u-complete] moleculeace: $run" >> "$LOG"
    ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
      --model "$run" --encoder "$enc" --tokenizer figure_data/_tokenizer \
      --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
    aws s3 cp --recursive "figure_data/chemeleon_suite/moleculeace/$run" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$run" --only-show-errors
  fi
done

done=0
for s in s0 s1 s2; do
  run=s2u_dense_from8M_$s
  [ -f "figure_data/climb_v2_phase2/$run/moleculenet_cv/moleculenet_summary.csv" ] \
    && [ -f "figure_data/chemeleon_suite/moleculeace/$run/verified.json" ] && done=$((done+1))
done
echo "[s2u-complete] complete=$done/3 $(date -u +%FT%TZ)" >> "$LOG"
if [ "$done" -eq 3 ]; then
  touch figure_data/S2U_COMPLETE_DONE
  echo "[s2u-complete] all done -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[s2u-complete] incomplete -> staying UP for inspection $(date -u +%FT%TZ)" >> "$LOG"
fi

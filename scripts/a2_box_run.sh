#!/usr/bin/env bash
# A2 error-bar bootstrap, run ON AWS so it survives the laptop closing. Pulls every input from S3.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis figure_data
LOG=analysis/a2_box.log
echo "[a2] start $(date -u +%FT%TZ)" >> "$LOG"
# inputs: per-molecule OOF (MolNet + CBS), MoleculeACE results, Polaris scores, aggregated tables
aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2 figure_data/climb_v2_phase2 \
  --exclude "*" --include "*/moleculenet_cv/test_predictions.csv" --include "*/moleculenet_cv/moleculenet_summary.csv" --only-show-errors
aws s3 sync s3://climb-s3-bucket/experiments/cbs_benchmark figure_data/cbs_benchmark \
  --exclude "*" --include "*/moleculenet_cv/test_predictions.csv" --include "*/moleculenet_cv/suite_summary.json" --include "*/moleculenet_cv/per_fold.csv" --only-show-errors
aws s3 sync s3://climb-s3-bucket/experiments/chemeleon_suite figure_data/chemeleon_suite \
  --exclude "*" --include "*.csv" --include "*.json" --only-show-errors
aws s3 sync s3://climb-s3-bucket/experiments/six_panel figure_data/six_panel --only-show-errors
aws s3 cp s3://climb-s3-bucket/experiments/analysis_rigor analysis/rigor --recursive --only-show-errors 2>/dev/null
echo "[a2] inputs staged $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/a2_bootstrap_errorbars.py >> "$LOG" 2>&1
rc=$?
echo "[a2] rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/six_panel/a2_errorbars.csv ]; then
  aws s3 cp figure_data/six_panel/a2_errorbars.csv s3://climb-s3-bucket/experiments/six_panel/a2_errorbars.csv --only-show-errors
  echo "[a2] uploaded -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[a2] NO OUTPUT -> staying UP" >> "$LOG"
fi

#!/bin/bash
# Wrapper to run the e2e label-efficiency fraction sweep on the GPU box, detached.
# Runs the driver, then (whatever the outcome) syncs results to S3 and records a
# provable completion summary derived from verified.json cells -- never from a bare
# file existing. Does NOT self-stop the instance; that stays the operator's call.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_labeleff_v2_frac_e2e
LOG=e2e_gpu.log
CELLDIR=figure_data/climb_v2_labeleff_v2_frac_e2e

echo "[wrap $(date -u +%H:%M:%S)] starting e2e fraction sweep on $(hostname)" | tee -a "$LOG"
$PY scripts/label_eff_fractions_e2e.py >> "$LOG" 2>&1
rc=$?
echo "[wrap $(date -u +%H:%M:%S)] driver exited rc=$rc" | tee -a "$LOG"

# provable completion: count verified cells (expect 91)
NVER=$(ls "$CELLDIR"/*/verified.json 2>/dev/null | wc -l | tr -d ' ')
echo "[wrap] verified cells = $NVER / 91" | tee -a "$LOG"

# durable backup regardless of outcome
aws s3 sync "$CELLDIR" "$S3" --only-show-errors || true
aws s3 cp analysis/rigor/label_efficiency_fractions_e2e.csv "$S3/_out/label_efficiency_fractions_e2e.csv" --only-show-errors || true
aws s3 cp analysis/rigor/label_efficiency_fractions_e2e_summary.csv "$S3/_out/label_efficiency_fractions_e2e_summary.csv" --only-show-errors || true
aws s3 cp "$LOG" "$S3/_out/$LOG" --only-show-errors || true

if [ "$NVER" -ge 91 ]; then
  echo "ALL_91_VERIFIED" | tee -a "$LOG"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$CELLDIR/_ALL_DONE"
  aws s3 cp "$CELLDIR/_ALL_DONE" "$S3/_ALL_DONE" --only-show-errors || true
else
  echo "INCOMPLETE ($NVER/91) -- box left up for inspection" | tee -a "$LOG"
fi
echo "E2E_GPU_WRAP_DONE rc=$rc verified=$NVER"

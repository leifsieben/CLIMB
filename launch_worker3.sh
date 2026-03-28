#!/usr/bin/env bash
# Auto-generated launch script for worker3 midpoint unsupervised scaling runs.
set -euo pipefail
mkdir -p /home/ec2-user/artifacts/robust_matrix_logs
cd /home/ec2-user/CLIMB

nohup /home/ec2-user/venvs/climb/bin/python scripts/launch_experiment_wave.py \
  --manifest /home/ec2-user/artifacts/robust_matrix/manifest.json \
  --run_id unsup_baseline_50000000_00 --run_id unsup_baseline_50000000_01 --run_id unsup_baseline_50000000_02 \
  --run_id unsup_baseline_250000000_00 --run_id unsup_baseline_250000000_01 --run_id unsup_baseline_250000000_02 \
  --run_id unsup_baseline_500000000_00 --run_id unsup_baseline_500000000_01 --run_id unsup_baseline_500000000_02 \
  --resume \
  --skip_existing \
  --worker_name worker3 \
  >> /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker3.log 2>&1 &

echo "Launched PID $!"
echo "Log: /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker3.log"

#!/usr/bin/env bash
# Auto-generated launch script for worker2
set -euo pipefail
mkdir -p /home/ec2-user/artifacts/robust_matrix_logs
cd /home/ec2-user/CLIMB

# Resume smoke if it exists (worker1), then run all assigned main runs
nohup /home/ec2-user/venvs/climb/bin/python scripts/launch_experiment_wave.py \
  --manifest /home/ec2-user/artifacts/robust_matrix/manifest.json \
  --run_id unsup_cov_10pct_10b --run_id unsup_cov_25pct_10b --run_id unsup_cov_50pct_10b --run_id unsup_cov_75pct_10b --run_id unsup_cov_100pct_10b --run_id mixed_10_90_00 --run_id mixed_10_90_01 --run_id mixed_10_90_02 --run_id mixed_20_80_00 --run_id mixed_20_80_01 --run_id mixed_20_80_02 --run_id mixed_50_50_00 --run_id mixed_50_50_01 --run_id mixed_50_50_02 --run_id mixed_80_20_00 --run_id mixed_80_20_01 --run_id mixed_80_20_02 --run_id mixed_90_10_00 --run_id mixed_90_10_01 --run_id mixed_90_10_02 \
  --resume \
  --skip_existing \
  --worker_name worker2 \
  >> /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker2.log 2>&1 &

echo "Launched PID $!"
echo "Log: /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker2.log"

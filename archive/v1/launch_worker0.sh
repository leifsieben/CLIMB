#!/usr/bin/env bash
# Auto-generated launch script for worker0
set -euo pipefail
mkdir -p /home/ec2-user/artifacts/robust_matrix_logs
cd /home/ec2-user/CLIMB

# Resume smoke if it exists (worker1), then run all assigned main runs
nohup /home/ec2-user/venvs/climb/bin/python scripts/launch_experiment_wave.py \
  --manifest /home/ec2-user/artifacts/robust_matrix/manifest.json \
  --run_id unsup_baseline_1000000_00 --run_id unsup_baseline_1000000_01 --run_id unsup_baseline_1000000_02 --run_id unsup_baseline_10000000_00 --run_id unsup_baseline_10000000_01 --run_id unsup_baseline_10000000_02 --run_id unsup_baseline_100000000_00 --run_id unsup_baseline_100000000_01 --run_id unsup_baseline_100000000_02 --run_id unsup_baseline_1000000000_00 --run_id unsup_baseline_1000000000_01 --run_id unsup_baseline_1000000000_02 --run_id unsup_baseline_10000000000_00 --run_id unsup_baseline_10000000000_01 --run_id unsup_baseline_10000000000_02 --run_id sup_order_1of5_00 --run_id sup_order_1of5_01 --run_id sup_order_1of5_02 --run_id sup_order_1of5_03 --run_id sup_order_1of5_04 \
  --resume \
  --skip_existing \
  --worker_name worker0 \
  >> /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker0.log 2>&1 &

echo "Launched PID $!"
echo "Log: /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker0.log"

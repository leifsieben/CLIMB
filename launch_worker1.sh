#!/usr/bin/env bash
# Auto-generated launch script for worker1
set -euo pipefail
mkdir -p /home/ec2-user/artifacts/robust_matrix_logs
cd /home/ec2-user/CLIMB

# Resume smoke if it exists (worker1), then run all assigned main runs
nohup /home/ec2-user/venvs/climb/bin/python scripts/launch_experiment_wave.py \
  --manifest /home/ec2-user/artifacts/robust_matrix/manifest.json \
  --run_id smoke_supervised_full_1epoch --run_id sup_order_2of5_00 --run_id sup_order_2of5_01 --run_id sup_order_2of5_02 --run_id sup_order_2of5_03 --run_id sup_order_2of5_04 --run_id sup_order_3of5_00 --run_id sup_order_3of5_01 --run_id sup_order_3of5_02 --run_id sup_order_3of5_03 --run_id sup_order_3of5_04 --run_id sup_order_4of5_00 --run_id sup_order_4of5_01 --run_id sup_order_4of5_02 --run_id sup_order_4of5_03 --run_id sup_order_4of5_04 --run_id sup_order_5of5_00 --run_id sup_order_5of5_01 --run_id sup_order_5of5_02 --run_id sup_order_5of5_03 --run_id sup_order_5of5_04 \
  --resume \
  --skip_existing \
  --worker_name worker1 \
  >> /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker1.log 2>&1 &

echo "Launched PID $!"
echo "Log: /home/ec2-user/artifacts/robust_matrix_logs/main_wave_worker1.log"

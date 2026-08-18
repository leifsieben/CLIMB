#!/usr/bin/env bash
# SI fig c: time the frozen CheMeleon fingerprint with the SAME harness/molecules/warm-up/repeats
# as every other row. GPU row on the A10G is the one that matters (the table's headline compares
# encoder-on-A10G vs the CPU ECFP4+desc anchor), so hardware_label matches the existing cuda rows.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis figure_data/_bench
LOG=analysis/bench_chemeleon.log
echo "[bench] start $(date -u +%FT%TZ)" >> "$LOG"
if [ ! -x ~/venvs/chemeleon/bin/python ]; then
  python3.12 -m venv ~/venvs/chemeleon
  ~/venvs/chemeleon/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  ~/venvs/chemeleon/bin/python -m pip install -q "chemprop==2.3.1" rdkit deepchem==2.5.0 xgboost >> "$LOG" 2>&1
  bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi
~/venvs/chemeleon/bin/python scripts/bench_featurization.py \
  --bench_chemeleon --skip_rdkit --skip_encoder \
  --devices cuda,cpu --cpu_threads 0,1 --n_molecules 1000 --repeats 5 \
  --hardware_label "AWS g5.2xlarge (NVIDIA A10G 24GB, 8 vCPU)" \
  --out figure_data/_bench/featurization_timing_chemeleon.json >> "$LOG" 2>&1
echo "[bench] rc=$? $(date -u +%FT%TZ)" >> "$LOG"
aws s3 cp figure_data/_bench/featurization_timing_chemeleon.json \
  s3://climb-s3-bucket/experiments/_bench/featurization_timing_chemeleon.json --only-show-errors
if [ -f figure_data/_bench/featurization_timing_chemeleon.json ]; then
  echo "[bench] done -> shutdown" >> "$LOG"; sudo shutdown -h now
else echo "[bench] incomplete -> staying UP" >> "$LOG"; fi

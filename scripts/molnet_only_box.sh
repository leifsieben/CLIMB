#!/usr/bin/env bash
# Dedicated MoleculeNet-only box: CheMeleon frozen probe + CheMeleon-foundation e2e on the 7
# MoleculeNet tasks (A1.b scaffold-5fold-CV). NO CBS work here (CBS runs on the main box). Runs
# detached; SELF-STOPS only when the e2e arm's verified.json proves all 7 datasets completed —
# a failure leaves the box UP for inspection (per aws-gpu-jobs). setsid+nohup → survives ssh.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/molnet_only.log
echo "[molnet-box] start $(date -u +%FT%TZ)" >> $LOG

# frozen CheMeleon on the 7 MoleculeNet tasks. Point CBS_CSV at a nonexistent file so
# chemeleon_bench.py skips its CBS arm (CBS is handled on the main box).
CBS_CSV=/home/ec2-user/CLIMB/data/__nocbs__.csv ~/venvs/chemeleon/bin/python scripts/chemeleon_bench.py >> $LOG 2>&1
echo "[molnet-box] frozen done $(date -u +%FT%TZ)" >> $LOG

# CheMeleon-foundation e2e (3 seeds, HIV last, incremental S3 sync per dataset)
~/venvs/chemeleon/bin/python scripts/molnet_chemprop_e2e.py >> $LOG 2>&1
echo "[molnet-box] MOLNET_ALL_DONE $(date -u +%FT%TZ)" >> $LOG

# gated self-stop: only if the e2e arm verified all 7 datasets
if [ -f figure_data/climb_v2_phase2/chemeleon_e2e/moleculenet_cv/verified.json ]; then
  echo "[molnet-box] verified -> stopping instance $(date -u +%FT%TZ)" >> $LOG
  sudo shutdown -h now
else
  echo "[molnet-box] NOT verified -> staying UP for inspection $(date -u +%FT%TZ)" >> $LOG
fi

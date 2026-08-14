#!/usr/bin/env bash
# Re-run the CheMeleon-foundation MoleculeNet e2e arm with PER-SEED dirs + per-molecule OOF dumps
# (the first run only saved aggregates, blocking the A1-table scaffold bootstrap). Detached; self-stops
# only when all 3 seed dirs verify. setsid+nohup -> survives ssh.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/molnet_rerun.log
echo "[molnet-rerun] start $(date -u +%FT%TZ)" >> $LOG

# env fix (idempotent): numpy<2 stack (tf 2.16) + deepchem ragged patch
bash scripts/molnet_box_bootstrap.sh >> $LOG 2>&1

# fresh: the old chemeleon_e2e dir held the averaged run with no OOF — remove so per-seed writes cleanly
rm -rf figure_data/climb_v2_phase2/chemeleon_e2e \
       figure_data/climb_v2_phase2/chemeleon_e2e_s1 \
       figure_data/climb_v2_phase2/chemeleon_e2e_s2

# 3 seeds x 7 datasets, scaffold-5fold, OOF dumped per dataset, HIV last; self-syncs per (seed,dataset)
~/venvs/chemeleon/bin/python scripts/molnet_chemprop_e2e.py >> $LOG 2>&1
echo "[molnet-rerun] MOLNET_RERUN_DONE $(date -u +%FT%TZ)" >> $LOG

V=figure_data/climb_v2_phase2
if [ -f $V/chemeleon_e2e/moleculenet_cv/verified.json ] && \
   [ -f $V/chemeleon_e2e_s1/moleculenet_cv/verified.json ] && \
   [ -f $V/chemeleon_e2e_s2/moleculenet_cv/verified.json ]; then
  echo "[molnet-rerun] all 3 seeds verified -> stopping $(date -u +%FT%TZ)" >> $LOG
  sudo shutdown -h now
else
  echo "[molnet-rerun] NOT all verified -> staying UP for inspection $(date -u +%FT%TZ)" >> $LOG
fi

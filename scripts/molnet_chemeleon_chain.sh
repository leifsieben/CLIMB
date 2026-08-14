#!/usr/bin/env bash
# Detached MoleculeNet CheMeleon chain: waits for the CBS chain to finish (frees the GPU), then runs
# the CheMeleon FROZEN fingerprint probe + the CheMeleon-foundation e2e arm on the 7 MoleculeNet tasks
# under the A1.b protocol (scaffold 5-fold CV), so both land in figure_data/climb_v2_phase2/ for the
# main figures. Idempotent sub-runners; each syncs to S3 as it goes. setsid+nohup → survives ssh.
set -u
cd /home/ec2-user/CLIMB
LOG=analysis/molnet_chemeleon.log
echo "[molnet-chain] start $(date -u +%FT%TZ)" >> $LOG

# 1. wait for CBS chain to finish (cap 4h). CBS_ALL_DONE is written to analysis/cbs_chain.log.
for i in $(seq 1 240); do
  grep -q CBS_ALL_DONE analysis/cbs_chain.log 2>/dev/null && { echo "[molnet-chain] CBS done" >> $LOG; break; }
  sleep 60
done

# 2. CheMeleon FROZEN probe on the 7 MoleculeNet tasks (scaffold 5-fold, 3 head seeds).
#    chemeleon_bench.py also re-checks CBS-frozen but skips it (already verified) — molnet is the target.
~/venvs/chemeleon/bin/python scripts/chemeleon_bench.py >> $LOG 2>&1
echo "[molnet-chain] frozen done $(date -u +%FT%TZ)" >> $LOG

# 3. CheMeleon-foundation e2e arm (3 seeds, HIV last). Self-syncs each dataset to S3.
~/venvs/chemeleon/bin/python scripts/molnet_chemprop_e2e.py >> $LOG 2>&1
echo "[molnet-chain] MOLNET_ALL_DONE $(date -u +%FT%TZ)" >> $LOG

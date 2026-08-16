#!/usr/bin/env bash
# MoleculeACE pretraining-seed top-up runner + gated self-shutdown. Scores the 22 _s1/_s2 mainline
# encoders on MoleculeACE (frozen). Shuts down ONLY when all 22 verify; failures stay up.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/mace_seedtopup.log
echo "[mace-topup-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/six_panel_mace_seedtopup.py >> "$LOG" 2>&1
echo "[mace-topup-wrapper] driver exit rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/MACE_SEEDTOPUP_DONE ]; then
  echo "[mace-topup-wrapper] DONE -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[mace-topup-wrapper] NOT done -> staying UP $(date -u +%FT%TZ)" >> "$LOG"
fi

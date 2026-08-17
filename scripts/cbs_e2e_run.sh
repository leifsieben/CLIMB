#!/usr/bin/env bash
# CBS e2e (best-two, provided 5-fold, 3 seeds, NEF1%) runner + gated self-shutdown.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/cbs_e2e.log
echo "[cbs-e2e-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/cbs_e2e.py >> "$LOG" 2>&1
echo "[cbs-e2e-wrapper] driver exit rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/CBS_E2E_DONE ]; then
  echo "[cbs-e2e-wrapper] DONE -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[cbs-e2e-wrapper] NOT done -> staying UP $(date -u +%FT%TZ)" >> "$LOG"
fi

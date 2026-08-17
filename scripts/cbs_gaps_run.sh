#!/usr/bin/env bash
# CBS-gaps runner + gated self-shutdown (4 mixed/minimol arms never run on CBS).
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/cbs_gaps.log
echo "[cbs-gaps-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/cbs_gaps.py >> "$LOG" 2>&1
echo "[cbs-gaps-wrapper] exit rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/CBS_GAPS_DONE ]; then
  echo "[cbs-gaps-wrapper] done -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[cbs-gaps-wrapper] incomplete -> staying UP $(date -u +%FT%TZ)" >> "$LOG"
fi

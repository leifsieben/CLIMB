#!/usr/bin/env bash
# hERG-on-scaling-encoders runner + gated self-shutdown (two-venv: CLIMB predict + polaris score).
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/six_panel_herg.log
echo "[herg-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/six_panel_herg.py >> "$LOG" 2>&1
echo "[herg-wrapper] exit rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/SIX_PANEL_HERG_DONE ]; then
  echo "[herg-wrapper] done -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[herg-wrapper] incomplete -> staying UP $(date -u +%FT%TZ)" >> "$LOG"
fi

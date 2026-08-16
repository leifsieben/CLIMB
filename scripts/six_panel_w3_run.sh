#!/usr/bin/env bash
# Wave-3 detached runner + gated self-shutdown. Fine-tunes the best-two encoders across the
# fraction grid on {BACE,BBBP,Tox21,QM7}. Shuts the box down ONLY if the completion marker is
# present (all cells done); any failure leaves the box UP for inspection.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/six_panel_w3.log
echo "[w3-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/six_panel_e2e.py >> "$LOG" 2>&1
rc=$?
echo "[w3-wrapper] driver exit rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/SIX_PANEL_W3_DONE ]; then
  echo "[w3-wrapper] DONE marker present -> shutdown $(date -u +%FT%TZ)" >> "$LOG"
  sudo shutdown -h now
else
  echo "[w3-wrapper] NO done marker -> staying UP for inspection $(date -u +%FT%TZ)" >> "$LOG"
fi

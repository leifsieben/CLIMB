#!/usr/bin/env bash
# Wave-2 detached runner + gated self-shutdown. Runs the frozen re-eval driver over all scaling
# encoders (MoleculeACE panel; CBS auto-skips while data/cbs.csv is absent). Shuts the box down
# ONLY if the completion marker is present (all encoders done) — any failure leaves the box UP for
# inspection, never terminating logs/partials. setsid+nohup at launch -> survives ssh disconnect.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/six_panel_w2.log
echo "[w2-wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
~/venvs/climb/bin/python scripts/six_panel_frozen_reeval.py >> "$LOG" 2>&1
rc=$?
echo "[w2-wrapper] driver exit rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/SIX_PANEL_W2_DONE ]; then
  echo "[w2-wrapper] DONE marker present -> shutdown $(date -u +%FT%TZ)" >> "$LOG"
  sudo shutdown -h now
else
  echo "[w2-wrapper] NO done marker -> staying UP for inspection $(date -u +%FT%TZ)" >> "$LOG"
fi

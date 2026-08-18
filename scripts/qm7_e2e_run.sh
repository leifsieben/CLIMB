#!/usr/bin/env bash
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/qm7_e2e_run.log
say() { echo "[qm7e2e] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
~/venvs/climb/bin/python scripts/qm7_native_e2e.py >> "$LOG" 2>&1
say "rc=$?"
if [ -f figure_data/QM7_NATIVE_E2E_DONE ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP for inspection"; fi

#!/usr/bin/env bash
# QM7 native-units re-eval, one shard per box. Gated: powers off only when every assigned run has
# QM7 fold rows in NATIVE units (RMSE ~200 kcal/mol, not the stale ~0.85 z-scored value).
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/qm7_run.log
say() { echo "[qm7run] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start shard ${SHARD:-0}/${NSHARD:-1}"
SHARD="${SHARD:-0}" NSHARD="${NSHARD:-1}" ~/venvs/climb/bin/python scripts/qm7_native_reeval.py >> "$LOG" 2>&1
say "rc=$?"
if [ -f "figure_data/QM7_NATIVE_DONE_${SHARD:-0}" ]; then
  say "COMPLETE -> shutdown"; sudo shutdown -h now
else
  say "INCOMPLETE -> staying UP for inspection"
fi

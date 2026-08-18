#!/usr/bin/env bash
# Shutdown-only supervisor: waits for this box's QM7 shard, verifies it completed, powers off.
# Used when a chained follow-on job has been MOVED to another box that freed up -- without this the
# box would either duplicate that job or sit idle with nobody owning shutdown.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/ladder_chain.log
say() { echo "[chain] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "shutdown-only supervisor (follow-on moved to another box)"
while pgrep -f "qm7_native_reeval.py" > /dev/null; do sleep 60; done
if [ -f "figure_data/QM7_NATIVE_DONE_${SHARD:-1}" ]; then
  say "shard complete -> shutdown"; sudo shutdown -h now
else
  say "shard INCOMPLETE -> staying UP for inspection"
fi

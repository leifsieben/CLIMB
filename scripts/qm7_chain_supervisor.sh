#!/usr/bin/env bash
# Runs the e2e QM7 re-eval AFTER this box finishes its frozen QM7 shard, then powers off once.
#
# Why chained rather than its own box: the account is at its 64 vCPU limit for this instance
# bucket, so StartInstances returns VcpuLimitExceeded -- there is no room for another GPU box
# until one of the running ones exits.
#
# This supervisor takes shutdown ownership from qm7_run.sh (whose wrapper is killed, leaving its
# python running) so the box cannot power off between the two jobs.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/qm7_chain.log
say() { echo "[chain] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "supervisor start (shard ${SHARD:-0})"

while pgrep -f "qm7_native_reeval.py" > /dev/null; do sleep 60; done
say "frozen QM7 shard finished"

FROZEN_OK=0
[ -f "figure_data/QM7_NATIVE_DONE_${SHARD:-0}" ] && FROZEN_OK=1
say "frozen shard complete=$FROZEN_OK"

say "starting e2e QM7 re-eval"
~/venvs/climb/bin/python scripts/qm7_native_e2e.py >> analysis/qm7_e2e_run.log 2>&1
E2E_OK=0
[ -f figure_data/QM7_NATIVE_E2E_DONE ] && E2E_OK=1
say "e2e complete=$E2E_OK"

if [ "$FROZEN_OK" = "1" ] && [ "$E2E_OK" = "1" ]; then
  say "all verified -> shutdown"; sudo shutdown -h now
else
  say "NOT shutting down (frozen=$FROZEN_OK e2e=$E2E_OK)"
fi

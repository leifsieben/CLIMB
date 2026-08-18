#!/usr/bin/env bash
# Chains a follow-on job onto a ladder box once its QM7 shard finishes, and owns shutdown for both.
# Needed because the account is at its vCPU limit / AWS has no spare GPU capacity, so the only way
# to start the next job is to inherit a box that is already running.
# NEXT_JOB = script to run after the shard completes.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/ladder_chain.log
say() { echo "[chain] $(date -u +%FT%TZ) $*" >> "$LOG"; }
NEXT_JOB="${NEXT_JOB:?set NEXT_JOB}"
say "supervisor start, will run $NEXT_JOB after the QM7 shard"
while pgrep -f "qm7_native_reeval.py" > /dev/null; do sleep 60; done
say "QM7 shard finished; starting $NEXT_JOB"
bash "scripts/$NEXT_JOB"        # each follow-on owns its own gate + shutdown
say "$NEXT_JOB returned $?"

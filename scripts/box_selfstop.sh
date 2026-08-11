#!/bin/bash
# Session-independent, cost-safe auto-stop for the Experiment-A box. Runs ON the box, detached.
# Waits for the bigram wave's verified-completion marker (EXPA_BIGRAM_DONE, written only after the
# driver confirms 3/3 verified + CV), then STOPS the instance. Instance shutdown behavior is 'stop'
# (verified), so EBS/encoders/logs are preserved; all eval results are already on S3.
#
# Deliberately does NOT stop on failure: if bigram never produces the marker (crash/stall), this
# watcher keeps waiting and the box stays up for inspection (per the aws-gpu-jobs discipline —
# never blind-shutdown; a failed run must leave its state behind).
set -uo pipefail
LOG=/home/ec2-user/synth/box_selfstop.log
MARK=/home/ec2-user/synth/expA_bigram_run.log
say(){ echo "[selfstop $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "armed: will stop the box after EXPA_BIGRAM_DONE"
while true; do
  if grep -q EXPA_BIGRAM_DONE "$MARK" 2>/dev/null; then
    n=$(grep -oE "done=[0-9]+/3" "$MARK" | tail -1)
    say "bigram complete ($n); results on S3; stopping instance in 3 min"
    bash /home/ec2-user/CLIMB/scripts/notify.sh DONE "ExpA box AUTO-STOP" \
        "bigram $n complete; stopping instance in 3 min. All evals on S3 under climb_v2_expA. Restart the box to resume." || true
    sleep 180
    sudo shutdown -h now
    exit 0
  fi
  sleep 300
done

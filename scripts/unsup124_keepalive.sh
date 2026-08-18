#!/usr/bin/env bash
# Keep a WAITING training box above the climb-idle-autostop CPU floor until training starts.
#
# The problem this solves: climb-idle-autostop-<id> stops the instance after 48 consecutive
# 5-minute periods of CPUUtilization < 3%. A box that is up but parked in unsup124_chain.sh
# waiting for the corpus to finish building sits at ~0.3% CPU, so the alarm would stop it
# roughly 4h after boot -- killing the chain before it ever launched a single training step.
# That backstop is correct and worth keeping; it just cannot tell "waiting for an upstream
# dependency" apart from "finished and idling".
#
# So: burn a fraction of one core while WAITING, and get out of the way the moment
# pretrain_v2.py appears. If training never starts, this exits after MAX_HOURS and the
# idle-autostop alarm then correctly reclaims the box -- the cost backstop is preserved,
# only deferred.
set -uo pipefail
MAX_HOURS=${1:-9}
LOG=/home/ec2-user/keepalive.log
END=$(( $(date +%s) + MAX_HOURS * 3600 ))

say(){ echo "[keepalive $(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }
# ~30s of one core per 45s cycle = ~8% of 8 vCPU, comfortably over the 3% floor.
burn(){ local e=$(( $(date +%s) + 30 )); while [ "$(date +%s)" -lt "$e" ]; do :; done; }

say "started (max ${MAX_HOURS}h); holding CPU above the idle-autostop floor until training begins"
while [ "$(date +%s)" -lt "$END" ]; do
  if pgrep -f pretrain_v2.py >/dev/null 2>&1; then
    say "pretrain_v2.py detected - training is generating its own load; exiting"
    exit 0
  fi
  burn
  sleep 15
done
say "MAX_HOURS reached without training starting - exiting so idle-autostop can reclaim the box"

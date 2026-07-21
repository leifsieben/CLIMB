#!/usr/bin/env bash
# Run the NEXT manifest on this box once the CURRENT worker exits — entirely on-box, so the
# hand-off does not depend on any laptop staying awake.
#
# The obstacle is that phase2_worker.sh self-stops the instance the moment its manifest is fully
# verified, which would take the box away before a follow-on wave could start. The caller
# therefore masks poweroff.target first, so that `sudo shutdown -h now` fails and the box
# survives; this script unmasks it again before starting the next wave, so THAT wave stops the
# box normally when it finishes and nothing is left billing.
#
# Masking is a safe bet rather than a gamble: if it works the box stays up and the next wave
# runs; if it somehow does not, the box merely stops exactly as it would have anyway.
#
# Usage: onbox_chain.sh <next_manifest> <next_worker_name> <pgrep_pattern_of_current_worker>
set -uo pipefail
NEXT_MANIFEST="${1:?usage: onbox_chain.sh <manifest> <worker> <wait_pattern>}"
NEXT_WORKER="${2:?}"
WAIT_PAT="${3:?}"
cd /home/ec2-user/CLIMB

echo "[chain] waiting for current worker matching '$WAIT_PAT' to exit ($(date -u))"
while pgrep -f "$WAIT_PAT" >/dev/null 2>&1; do sleep 30; done
echo "[chain] current worker exited ($(date -u))"

# Restore normal shutdown BEFORE launching, so the next wave can stop the box when it is done.
sudo systemctl unmask poweroff.target >/dev/null 2>&1 || true
sudo systemctl unmask halt.target     >/dev/null 2>&1 || true
echo "[chain] poweroff.target unmasked; launching $NEXT_WORKER"

bash scripts/phase2_worker.sh "$NEXT_MANIFEST" "$NEXT_WORKER" > "phase2_${NEXT_WORKER}.log" 2>&1
echo "[chain] $NEXT_WORKER finished with rc=$? ($(date -u))"

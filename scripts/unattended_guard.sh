#!/usr/bin/env bash
# Run a wave worker with NOBODY watching for two weeks.
#
# The normal worker deliberately stays up when a run fails, so the box can be inspected. That is
# right when someone is around and catastrophic when nobody is: an idle g5.2xlarge costs ~$1.2/h,
# so a failure on day 1 of a fortnight's absence burns ~$400 per box while achieving nothing.
#
# This wrapper resolves that without giving up diagnosability: on EVERY exit path -- success,
# failure, hang, or deadline -- it first pushes the entire wave directory AND the logs to S3, then
# alerts, then stops the instance. Nothing is lost by stopping, because everything needed to
# diagnose is already in S3 before the shutdown call is made.
#
#   Usage: unattended_guard.sh <manifest.json> <tag> [max_hours]
#
# Stop, not terminate: stopping ends all GPU charges (the actual concern) while preserving the
# root volume, the venv and the local tree, so work can resume with `aws ec2 start-instances`
# instead of a rebuild. Terminating would save ~$8/box over two weeks and cost a full re-setup.
set -uo pipefail
cd /home/ec2-user/CLIMB

MANIFEST=${1:?manifest required}
TAG=${2:?tag required}
MAX_HOURS=${3:-20}
LOG=phase2_${TAG}.log
GUARD_LOG=guard_${TAG}.log
S3ROOT=$(python3 -c "import json;print(json.load(open('$MANIFEST')).get('s3_backup_root',''))")
LOCALROOT=$(python3 -c "import json;print(json.load(open('$MANIFEST')).get('results_root',''))")

say(){ echo "[guard $(date -u +%H:%M:%S)] $*" | tee -a "$GUARD_LOG"; }

save_everything() {
    say "pushing results + logs to S3 before doing anything irreversible"
    # Whole wave dir. Tokenizers are excluded (they are identical per run and already in S3);
    # nothing else is, because at this point we do not know what will turn out to matter.
    [ -n "$LOCALROOT" ] && [ -d "$LOCALROOT" ] && \
        aws s3 sync "$LOCALROOT" "$S3ROOT" --exclude "*/tokenizer/*" --only-show-errors
    for f in "$LOG" "$GUARD_LOG" "$MANIFEST"; do
        [ -f "$f" ] && aws s3 cp "$f" "$S3ROOT/_logs/$(basename "$f")" --only-show-errors
    done
    say "S3 push complete"
}

trap 'say "guard received a signal"; save_everything' INT TERM

say "starting worker: manifest=$MANIFEST tag=$TAG deadline=${MAX_HOURS}h"
setsid bash scripts/phase2_worker.sh "$MANIFEST" "$TAG" > "$LOG" 2>&1 < /dev/null &
WPID=$!
say "worker pid=$WPID"

DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 ))
while kill -0 "$WPID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        say "DEADLINE of ${MAX_HOURS}h reached - killing the worker and saving state"
        kill -TERM "$WPID" 2>/dev/null; sleep 60; kill -KILL "$WPID" 2>/dev/null
        save_everything
        bash scripts/notify.sh ALERT "[$TAG] hit the ${MAX_HOURS}h deadline" \
            "Worker killed at its deadline. Everything is in $S3ROOT (results) and $S3ROOT/_logs/. Instance stopping."
        sudo shutdown -h now; exit 3
    fi
    sleep 120
done
wait "$WPID"; RC=$?
say "worker exited rc=$RC"

save_everything

# Report what actually completed, from the completion markers rather than from the exit code:
# a worker can exit 0 having skipped everything.
NRUNS=$(python3 -c "import json;m=json.load(open('$MANIFEST'));print(len(m['runs']))")
NDONE=$(python3 - <<PY
import json,os
m=json.load(open("$MANIFEST"))
print(sum(os.path.exists(os.path.join(r["output_dir"],"verified.json")) for r in m["runs"]))
PY
)
say "verified complete: $NDONE / $NRUNS"

if [ "$RC" -eq 0 ] && [ "$NDONE" -eq "$NRUNS" ]; then
    bash scripts/notify.sh DONE "[$TAG] all $NRUNS runs verified complete" \
        "Results in $S3ROOT. Instance stopping."
else
    bash scripts/notify.sh ALERT "[$TAG] FINISHED INCOMPLETE: $NDONE/$NRUNS verified (rc=$RC)" \
        "Everything (partial checkpoints, metrics, logs) is already in $S3ROOT and $S3ROOT/_logs/ - nothing is lost by the stop. Restart with: aws ec2 start-instances --instance-ids <id>"
fi

say "stopping the instance"
sudo shutdown -h now

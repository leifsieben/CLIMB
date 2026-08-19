#!/usr/bin/env bash
# Re-eval Tox21 for the 20 runs whose PREDICTIONS predate the 2026-08-05 masking fix and therefore
# cannot be repaired from disk. Writes moleculenet_cv_tox21fixed/ only; never touches the original.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/tox21_reeval_run.log
say() { echo "[t21run] $(date -u +%FT%TZ) $*" >> "$LOG"; }
SHARD="${SHARD:-0}"; NSHARD="${NSHARD:-1}"
say "start shard $SHARD/$NSHARD"

# The whole job is worthless if the box's code predates the fix, so PROVE it is present first.
~/venvs/climb/bin/python - <<'PY' >> "$LOG" 2>&1 || { say "FATAL masking fix absent on this box -> staying UP"; exit 1; }
import sys; sys.path.insert(0,'.')
import inspect, eval_v2
src = inspect.getsource(eval_v2._load_moleculenet)
assert "_y_masked" in src or "w==0" in src or "w == 0" in src, "masking fix NOT in eval_v2 on this box"
print("[t21run] masking fix verified present in eval_v2._load_moleculenet")
PY

[ -s /tmp/reeval20.txt ] || { say "FATAL no run list"; exit 1; }
SHARD=$SHARD NSHARD=$NSHARD ~/venvs/climb/bin/python scripts/tox21_reeval_prefix_runs.py \
  --list /tmp/reeval20.txt --shard "$SHARD" --nshard "$NSHARD" >> "$LOG" 2>&1
say "rc=$?"
if [ -f "figure_data/TOX21_REEVAL_DONE_${SHARD}" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP for inspection"; fi

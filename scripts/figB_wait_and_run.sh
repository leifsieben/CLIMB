#!/usr/bin/env bash
# Wait until a rung's descriptor prerequisites are ACTUALLY complete, then run it.
#
# A box tagged "standby" was standing by in name only: no waiter, no poller, nothing that would
# start the rung when its shards landed. The shards would have completed into an empty room. A
# handoff nobody owns is not a handoff, so it gets a process.
#
# The wait condition is verify_descriptor_dir's own budget-scoped check -- the SAME gate the run
# itself uses -- so the waiter cannot decide the data is ready on a rule the run disagrees with.
set -u
set -o pipefail
RUN=$1
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
LOG=analysis/wait_${RUN}.log
mkdir -p analysis
say () { echo "[wait:$RUN] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

say "waiting for the descriptor prerequisites of $RUN"
# Build the manifest once, just to read the budget the rung will actually use.
aws s3 cp "s3://climb-s3-bucket/experiments/climb_v2_phase2/manifests/templates/manifest.json" \
  experiments/climb_v2_phase2/manifest.json --only-show-errors || { say "FATAL template fetch"; exit 1; }
aws s3 cp "s3://climb-s3-bucket/experiments/climb_v2_phase2/manifests/templates/manifest_supplement.json" \
  experiments/climb_v2_phase2/manifest_supplement.json --only-show-errors || { say "FATAL supplement fetch"; exit 1; }
$PY scripts/build_figB_manifest.py --run "$RUN" --out "analysis/manifest_${RUN}.json" >/dev/null 2>&1 \
  || { say "FATAL manifest builder refused"; exit 1; }
BUDGET=$($PY -c "
import json; print(json.load(open('analysis/manifest_${RUN}.json'))['runs'][0]['selection']['total_forward_passes'])") \
  || { say "FATAL cannot read the budget"; exit 1; }
say "budget $BUDGET forward passes"

for i in $(seq 1 480); do   # 480 x 60s = 8h ceiling on WAITING, not on the run
  if $PY scripts/verify_descriptor_dir.py --corpus pubchem_124m_full --budget "$BUDGET" >/dev/null 2>&1; then
    say "descriptor directory complete for $BUDGET forward passes -- launching"
    exec bash scripts/figB_run.sh "$RUN"
  fi
  [ $((i % 5)) -eq 0 ] && say "still waiting ($i min): $($PY scripts/verify_descriptor_dir.py --corpus pubchem_124m_full --budget "$BUDGET" 2>&1 | grep -E 'complete' | tail -1)"
  sleep 60
done
say "GAVE UP after 8h of waiting -- BOX STAYS UP"
exit 1

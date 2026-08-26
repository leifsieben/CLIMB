#!/usr/bin/env bash
# Run ONE fig_B rung end to end on a fresh box, then self-terminate on verified completion.
#
# Usage: figB_run.sh <run_id>
#
# The box is launched with instance-initiated-shutdown-behavior=terminate, so `shutdown -h` is all
# this needs and the instance role stays S3-only.
set -u
RUN=$1
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
LOG=analysis/figB_${RUN}.log
mkdir -p analysis
say () { echo "[figB:$RUN] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors; exit 1; }

say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux
say "code at $(git rev-parse --short HEAD)"

# ---- manifest, gated ------------------------------------------------------------------------
$PY scripts/build_figB_manifest.py --run "$RUN" --out "analysis/manifest_${RUN}.json" 2>&1 | tee -a "$LOG"
[ -s "analysis/manifest_${RUN}.json" ] || abort "manifest builder refused or failed"

# ---- assert the corpus is reachable BEFORE spending hours -------------------------------------
# A missing or misnamed corpus prefix returns empty and exits 0 from `aws s3 ls`, so COUNT.
n=$(aws s3 ls s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full/ | grep -c "\.parquet$")
[ "$n" -ge 124 ] || abort "expected >=124 corpus shards, counted $n"
say "corpus OK -- $n shards"
# And the one field whose presence would silently corrupt MTR on this corpus.
$PY -c "
import json
m=json.load(open('analysis/manifest_${RUN}.json'))
pc=m['runs'][0]['pretrain_config']
assert 'descriptor_precompute_dir' not in pc, 'descriptor_precompute_dir is set -- shard names collide across corpora'
print('[figB] descriptor_precompute_dir correctly absent')" 2>&1 | tee -a "$LOG" || abort "descriptor precompute check failed"

# ---- train ------------------------------------------------------------------------------------
say "launching wave"
$PY scripts/launch_v2_wave.py --manifest "analysis/manifest_${RUN}.json" --worker_name "figB_${RUN}" \
  >> "$LOG" 2>&1
rc=$?
say "wave exited rc=$rc"

# ---- completion is ACHIEVED forward passes, never a file ---------------------------------------
d=experiments/climb_v2_phase2/$RUN
want=$($PY -c "
import json; print(json.load(open('analysis/manifest_${RUN}.json'))['runs'][0]['selection']['total_forward_passes'])")
got=$($PY -c "
import json
try:
    print(json.loads(open('$d/metrics.jsonl').read().strip().split(chr(10))[-1])['forward_passes_seen'])
except Exception: print(0)" 2>/dev/null)
say "forward passes $got / $want"
$PY -c "import sys; sys.exit(0 if $got >= 0.98*$want else 1)" || abort "only $got of $want forward passes -- truncated"

# ---- upload BEFORE the gates, so a failing gate still leaves the work durable -------------------
aws s3 cp "$d" "$S3/$RUN" --recursive --only-show-errors || abort "upload failed"
aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors
say "uploaded"

# ---- reconcile against S3 rather than trusting the local state ----------------------------------
for f in encoder/model.safetensors metrics.jsonl config.yaml; do
  aws s3 ls "$S3/$RUN/$f" >/dev/null 2>&1 || abort "$f absent on S3 after upload"
done
say "ALL GATES PASSED -- terminating"
aws s3 cp "$LOG" "$S3/$RUN/figB_run.log" --only-show-errors
sudo shutdown -h now

#!/usr/bin/env bash
# Attack a SMALL list of blocking shards with every core on the box, splitting each shard by ROWS.
#
# precompute_c124_box.sh pins one shard per core, so a rung blocked on five named shards waits
# ~2.9h however big the box is. Here the five shards are split into row parts, every core takes a
# part, and the shards land in ~cores/shards times less wall clock. Used only for the critical
# path -- for bulk work the per-shard runner is simpler and just as fast in aggregate.
set -u
set -o pipefail
LIST=$1              # comma-separated 5-digit shard ids
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
CORPUS=pubchem_124m_full
OUT=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_descriptors
LOG=analysis/precompute_split.log
mkdir -p analysis
say () { echo "[split] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$OUT/_logs/split_box.log" --only-show-errors; exit 1; }

CORES=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print (c*s>0)?c*s:0}')
[ "${CORES:-0}" -gt 0 ] || CORES=$(nproc)
N=$(echo "$LIST" | tr ',' '\n' | grep -c .)
say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown), $CORES cores, $N shards: $LIST"

# ---- the descriptor set must be the fleet's -----------------------------------------------------
if $PY -m pip list 2>/dev/null | grep -q "^rdkit-pypi"; then
  say "rdkit-pypi present -- repairing to the fleet rdkit"
  $PY -m pip uninstall -y rdkit-pypi >/dev/null 2>&1
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" >/dev/null 2>&1
fi
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors \
  || abort "cannot fetch the canonical descriptor stats"
$PY -c "
import json, descriptors_v2 as dv
stats = json.load(open('configs/descriptor_stats.json'))
cur = dv.descriptor_names()
assert cur == stats['names'], f'rdkit exposes {len(cur)} names against {len(stats[\"names\"])} in the stats, or the order differs'
print(f'[split] descriptor set matches the canonical stats: {len(cur)} names in order')" 2>&1 | tee -a "$LOG" \
  || abort "descriptor set does not match the canonical stats"

# ---- prove the ROW-SLICED path reproduces the per-shard path, on bytes already published ---------
# This is new code writing the array every MTR run trains against, and a row-offset bug produces a
# perfectly well-formed file of wrong rows. So before computing anything we cannot check, recompute
# a slice of a shard that is ALREADY on S3 and require it to match exactly.
say "self-test: recomputing a published slice and comparing bytes"
$PY scripts/selftest_row_slice.py --corpus "$CORPUS" --exclude "$LIST" 2>&1 | tee -a "$LOG" \
  || abort "row-sliced descriptors do NOT reproduce the published shard -- refusing to write"

# ---- one part per core, all shards in flight -----------------------------------------------------
PARTS=$(( CORES / N )); [ "$PARTS" -lt 1 ] && PARTS=1
say "splitting each shard into $PARTS row parts ($((PARTS * N)) processes)"
$PY scripts/plan_row_parts.py --corpus "$CORPUS" --shards "$LIST" --parts "$PARTS" > analysis/row_jobs.txt \
  || abort "could not plan the row parts"
say "planned $(wc -l < analysis/row_jobs.txt) parts"

xargs -P "$((PARTS * N))" -I{} sh -c \
  "$PY scripts/precompute_rows.py --corpus $CORPUS {} >> analysis/rows_\$(echo {} | tr ' /-' '___').log 2>&1 || echo FAILED {} >> analysis/row_failures.txt" \
  < analysis/row_jobs.txt
say "part pool drained"
[ -s analysis/row_failures.txt ] && abort "parts failed: $(tr '\n' ';' < analysis/row_failures.txt)"

# ---- merge; the merger refuses unless the parts tile the shard exactly ---------------------------
for s in $(echo "$LIST" | tr ',' ' '); do
  idx=$(echo "$s" | sed 's/^0*//;s/^$/0/')
  $PY scripts/merge_shard_parts.py --corpus "$CORPUS" --shard "$idx" 2>&1 | tee -a "$LOG" \
    || abort "merge failed for shard $s"
done

# ---- completion is the object's size against the writer's own row count --------------------------
$PY scripts/verify_descriptor_dir.py --corpus "$CORPUS" --shards "$LIST" 2>&1 | tee -a "$LOG" \
  || abort "post-merge verification failed"

say "ALL $N SHARDS VERIFIED -- uploading log"
aws s3 cp "$LOG" "$OUT/_logs/split_box.log" --only-show-errors
# A box that still owes other work must not terminate on finishing this list. Opt-in rather than
# opt-out: the failure that costs a night is a box that shut down with work still assigned, not
# one that idled for ten minutes while the next stage was dispatched.
if [ "${SPLIT_SHUTDOWN:-0}" = "1" ]; then
  say "SPLIT_SHUTDOWN=1 -- terminating"
  sudo shutdown -h now
else
  say "SPLIT_SHUTDOWN unset -- staying up for the next stage"
fi

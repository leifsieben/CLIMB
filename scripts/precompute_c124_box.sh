#!/usr/bin/env bash
# Precompute the 217 rdkit descriptors for an explicit LIST of pubchem_124m_full shards, in parallel
# across every physical core, verify, and self-terminate.
#
# Usage: precompute_c124_box.sh <label> <comma-separated shard ids>
#
# A LIST rather than a range, and one process per PHYSICAL core rather than per vCPU, because both
# choices decide when training can start. The list is dealt out in the order a rung actually opens
# its shards (scripts/c124_priority_order.py), so the 50M rung's 72 shards land before the 52 only
# the 100M rung needs. Running 2x more processes than cores does not increase throughput -- it just
# means nothing finishes until everything finishes, which is the opposite of what a priority order
# is for.
set -u
LABEL=$1; LIST=$2
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
OUT=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_descriptors
LOG=analysis/precompute_c124_${LABEL}.log
mkdir -p analysis
CORES=$(lscpu -p=CORE 2>/dev/null | grep -vc '^#' || nproc)
CORES=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print (c*s>0)?c*s:0}')
[ "${CORES:-0}" -gt 0 ] || CORES=$(nproc)
say () { echo "[pre:$LABEL] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$OUT/_logs/box_${LABEL}.log" --only-show-errors; exit 1; }

N=$(echo "$LIST" | tr ',' '\n' | grep -c .)
say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown), $CORES physical cores, $N shards"

# ---- the descriptor set must be the fleet's, or every byte written here is wrong ----------------
# The April AMI ships rdkit-pypi 2022.9.5 shadowing rdkit 2025.9.2 and exposes 208 of 217. Repair
# first, then check NAMES IN ORDER -- a reordered list of the right length would write a
# correct-shaped array of wrong columns and raise nothing.
if $PY -m pip list 2>/dev/null | grep -q "^rdkit-pypi"; then
  say "rdkit-pypi present -- repairing to the fleet rdkit"
  $PY -m pip uninstall -y rdkit-pypi >/dev/null 2>&1
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" >/dev/null 2>&1
fi
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors   || abort "cannot fetch the canonical descriptor stats"
$PY -c "
import json, descriptors_v2 as dv
stats = json.load(open('configs/descriptor_stats.json'))
cur = dv.descriptor_names()
assert cur == stats['names'], f'rdkit exposes {len(cur)} names against {len(stats[\"names\"])} in the stats, or the order differs'
print(f'[pre] descriptor set matches the canonical stats: {len(cur)} names in order')" 2>&1 | tee -a "$LOG"   || abort "descriptor set does not match the canonical stats"

# ---- one shard per physical core, IN THE GIVEN ORDER --------------------------------------------
say "precomputing $N shards across $CORES processes, priority order"
echo "$LIST" | tr ',' '\n' | grep . | xargs -P "$CORES" -I{} sh -c \
  "$PY scripts/precompute_descriptors.py --corpus pubchem_124m_full --shard_range \$(echo {} | sed 's/^0*//;s/^\$/0/')-\$(echo {} | sed 's/^0*//;s/^\$/0/') >> analysis/pre_shard_{}.log 2>&1 || echo FAILED {} >> analysis/pre_failures.txt"
say "worker pool drained"

# ---- completion is the OBJECT'S SIZE against the row count the WRITER reported -------------------
$PY - <<PYEOF 2>&1 | tee -a "$LOG"
import subprocess, sys, re, pathlib
ids = [x for x in "$LIST".split(",") if x.strip()]
out = subprocess.run(["aws", "s3", "ls", "$OUT/"], capture_output=True, text=True).stdout
got = {}
for line in out.splitlines():
    p = line.split()
    if len(p) >= 4 and p[-1].endswith(".npy"):
        got[p[-1]] = int(p[-2])
written = {}
for i in ids:
    f = pathlib.Path("analysis/pre_shard_%s.log" % i)
    if f.exists():
        m = re.findall(r"DONE shard_(\d+): wrote \((\d+), (\d+)\)", f.read_text())
        if m:
            written[m[-1][0]] = int(m[-1][1])
WIDTH, ITEMSIZE, HEADER = 217, 2, 128
bad = []
for i in ids:
    name = "descriptors_shard_%s.npy" % i
    if name not in got:
        bad.append((name, "ABSENT")); continue
    rows = written.get(i)
    if rows is None:
        bad.append((name, "no DONE line -- the writer never reported a shape")); continue
    want = rows * WIDTH * ITEMSIZE + HEADER
    if got[name] != want:
        bad.append((name, "%d bytes, expected %d for %d rows" % (got[name], want, rows)))
print("[pre] %d of %d shards complete" % (len(ids) - len(bad), len(ids)))
for n, w in bad[:20]:
    print("[pre] BAD %s: %s" % (n, w))
sys.exit(1 if bad else 0)
PYEOF
[ "${PIPESTATUS[0]}" = 0 ] || abort "not every assigned shard landed on S3"

# ---- verify BY MOLECULE, not by path -------------------------------------------------------------
# A correct path and a correct size are compatible with descriptors belonging to other molecules.
$PY scripts/verify_descriptor_alignment.py --corpus pubchem_124m_full --shard_list "$LIST" --n_probes 12 2>&1 | tee -a "$LOG"   || abort "BY-MOLECULE verification failed -- descriptors do not match their molecules"

aws s3 cp "$LOG" "$OUT/_logs/box_${LABEL}.log" --only-show-errors
say "ALL GATES PASSED -- terminating"
sudo shutdown -h now

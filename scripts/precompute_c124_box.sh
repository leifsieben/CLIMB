#!/usr/bin/env bash
# Precompute the 217 rdkit descriptors for a range of pubchem_124m_full shards, in parallel across
# every core, then verify BY MOLECULE and self-terminate.
#
# Usage: precompute_c124_box.sh <lo> <hi>
#
# Embarrassingly parallel: one shard per process, nproc processes at a time. The work is ~11 ms per
# molecule single-core and a shard is 1M molecules, so a shard is ~3 core-hours and the whole
# corpus ~377.
set -u
LO=$1; HI=$2
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
OUT=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_descriptors
LOG=analysis/precompute_c124_${LO}_${HI}.log
mkdir -p analysis
say () { echo "[pre:$LO-$HI] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$OUT/_logs/box_${LO}_${HI}.log" --only-show-errors; exit 1; }

say "start on $(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo unknown), $(nproc) cores"

# ---- the descriptor set must be the fleet's, or every byte written here is wrong ----------------
# The April AMI ships rdkit-pypi 2022.9.5 shadowing rdkit 2025.9.2 and exposes 208 of 217. Repair
# before checking, then check -- and check NAMES IN ORDER, because a reordered list of the right
# length would write a correct-shaped array of wrong columns and nothing would raise.
$PY -c "import rdkit" 2>/dev/null || true
$PY -m pip list 2>/dev/null | grep -q "^rdkit-pypi" && {
  say "rdkit-pypi present -- repairing to the fleet rdkit"
  $PY -m pip uninstall -y rdkit-pypi >/dev/null 2>&1
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" >/dev/null 2>&1
}
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors   || abort "cannot fetch the canonical descriptor stats"
$PY -c "
import json, descriptors_v2 as dv
stats = json.load(open('configs/descriptor_stats.json'))
cur = dv.descriptor_names()
assert cur == stats['names'], f'rdkit exposes {len(cur)} names, stats has {len(stats[\"names\"])}, or the order differs'
print(f'[pre] descriptor set matches the canonical stats: {len(cur)} names in order')" 2>&1 | tee -a "$LOG"   || abort "descriptor set does not match the canonical stats"

# ---- one shard per process, every core busy -----------------------------------------------------
say "precomputing shards $LO..$HI across $(nproc) processes"
seq "$LO" "$HI" | xargs -P "$(nproc)" -I{} sh -c \
  "$PY scripts/precompute_descriptors.py --corpus pubchem_124m_full --shard_range {}-{} >> analysis/pre_shard_{}.log 2>&1 || echo FAILED {} >> analysis/pre_failures.txt"
say "worker pool drained"

# ---- completion is the OBJECT'S SIZE, never the absence of an error ------------------------------
# 1,000,000 rows x 217 float16 + a 128-byte npy header = 434,000,128 bytes exactly. A truncated or
# short-written shard has a different size; a missing one has none.
$PY - <<PYEOF 2>&1 | tee -a "$LOG"
import subprocess, sys
lo, hi = $LO, $HI
def listing(prefix):
    out = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and not line.strip().startswith("PRE"):
            d[p[-1]] = int(p[-2])
    return d

got = listing("$OUT/")

import re, pathlib
written = {}
for i in range(lo, hi + 1):
    f = pathlib.Path("analysis/pre_shard_%d.log" % i)
    if not f.exists():
        continue
    m = re.findall(r"DONE shard_(\d+): wrote \((\d+), (\d+)\)", f.read_text())
    if m:
        written[int(m[-1][0])] = int(m[-1][1])
WIDTH, ITEMSIZE, HEADER = 217, 2, 128
bad = []
for i in range(lo, hi + 1):
    name = "descriptors_shard_%05d.npy" % i
    if name not in got:
        bad.append((name, "ABSENT")); continue
    # The last shard is short (389,701 rows), so size cannot be a single constant: take the row
    # count the WRITER reported for this shard and check S3 against that. This cross-checks the
    # process against the object rather than checking the object against a guess.
    rows = written.get(i)
    if rows is None:
        bad.append((name, "no DONE line in the worker log -- writer never reported a shape")); continue
    want = rows * WIDTH * ITEMSIZE + HEADER
    if got[name] != want:
        bad.append((name, "size %d != expected %d for %d rows" % (got[name], want, rows)))
print("[pre] shards on S3: %d of %d" % (hi - lo + 1 - len([b for b in bad if b[1] == "ABSENT"]), hi - lo + 1))
for name, why in bad:
    print("[pre] BAD %s: %s" % (name, why))
sys.exit(1 if bad else 0)
PYEOF
[ "${PIPESTATUS[0]}" = 0 ] || abort "not every shard in $LO..$HI landed on S3"

# ---- verify BY MOLECULE, not by path -------------------------------------------------------------
# A correct-looking path is perfectly compatible with descriptors attached to the wrong molecules.
# Recompute a sample live and assert against what the directory holds at the SAME row positions;
# misalignment comes back as noise, not as a match.
$PY scripts/verify_descriptor_alignment.py --corpus pubchem_124m_full --shards "$LO-$HI" --n_probes 24 2>&1 | tee -a "$LOG"   || abort "BY-MOLECULE verification failed -- descriptors do not match their molecules"

aws s3 cp "$LOG" "$OUT/_logs/box_${LO}_${HI}.log" --only-show-errors
say "ALL GATES PASSED -- terminating"
sudo shutdown -h now

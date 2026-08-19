#!/usr/bin/env bash
# ASK 4 (fig_C1's x-axis) in a numpy-2 venv.
#
# WHY: dedup_i1_reanalysis.py uses np.bitwise_count, which needs numpy >= 2.0. The climb venv is
# numpy 1.23.5, so it ran on the byte-lookup FALLBACK -- correct (verified bit-identical) but far
# slower, because the fallback materialises a per-byte uint16 array per comparison. Measured ETA on
# the fallback: ~45 HOURS for 12 shards. Native bitwise_count restores the fast path.
# This venv needs only numpy/pandas/pyarrow/rdkit -- no torch, so it builds in a couple of minutes.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/dedup_i1_np2.log
say() { echo "[i1np2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
PY=~/venvs/np2/bin/python
if ! $PY -c "import numpy,pandas,rdkit; assert numpy.__version__[0]>='2'" 2>/dev/null; then
  say "building numpy-2 venv"
  python3.12 -m venv ~/venvs/np2
  $PY -m pip install -q --upgrade pip wheel >> "$LOG" 2>&1
  $PY -m pip install -q "numpy>=2" pandas pyarrow rdkit >> "$LOG" 2>&1
fi
$PY -c "import numpy; assert hasattr(numpy,'bitwise_count'); print('[i1np2] numpy',numpy.__version__,'native bitwise_count OK')" >> "$LOG" 2>&1 \
  || { say "FATAL numpy-2 venv unusable -> staying UP"; exit 1; }

# stop the slow fallback run before starting the fast one, so they cannot race on the same output
pkill -f "python scripts/dedup_i1_reanalysis.py" 2>/dev/null && say "stopped the numpy-1 fallback run"
sleep 2

I1_TASKS="QM7 MoleculeACE" $PY scripts/dedup_i1_reanalysis.py --mode full >> "$LOG" 2>&1
say "rc=$?"
OUT=analysis/dedup_i1/full_corpus_similarity_i1.csv
if [ -s "$OUT" ] && awk -F, '$2=="MoleculeACE"' "$OUT" | head -1 | grep -q .; then
  aws s3 cp --recursive analysis/dedup_i1 s3://climb-s3-bucket/experiments/analysis_rigor/dedup_i1 --only-show-errors
  say "COMPLETE ($(awk -F, '$2=="MoleculeACE"' "$OUT" | wc -l) MoleculeACE rows) -> shutdown"
  sudo shutdown -h now
else
  say "INCOMPLETE -> staying UP"
fi

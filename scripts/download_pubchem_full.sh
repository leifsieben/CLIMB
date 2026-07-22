#!/usr/bin/env bash
# Download the FULL PubChem-124M corpus, not the 12M slice we have been training on.
#
# The existing corpus is 12 shards x 1,000,000 rows because
# preparing_datasets/prepare_pubchem_124m.py was invoked with `--rows 12000000` -- an explicit
# cap, not a filter and not a truncation bug. The upstream dataset
# (hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC) holds ~124M molecules.
#
# Writes to a NEW prefix so the 12M corpus every published run was trained on stays byte-identical
# and reproducible. Shards are uploaded incrementally, so an interruption loses at most one shard
# and the job can resume by re-running (completed shards are skipped).
#
# Re-canonicalisation is kept ON for consistency with the 12M corpus and the tokenizer that was
# fitted on it; it is also the throughput bottleneck (~2-5k mol/s), so expect many hours.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
OUT=raw_data/pubchem_124m_full
S3=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full
ROWS=${ROWS:-124000000}

mkdir -p "$OUT"
echo "[dl] starting $(date -u); target rows=$ROWS -> $OUT"
$PY preparing_datasets/prepare_pubchem_124m.py \
    --output-dir "$OUT" --rows "$ROWS" --shard-size 1000000 --recanonicalize
rc=$?
echo "[dl] prepare exited rc=$rc ($(date -u)); shards=$(ls "$OUT" 2>/dev/null | wc -l)"

aws s3 sync "$OUT" "$S3" --only-show-errors
echo "[dl] uploaded to $S3"
bash scripts/notify.sh "$([ $rc -eq 0 ] && echo DONE || echo ALERT)" \
  "PubChem full download rc=$rc ($(ls "$OUT" | wc -l) shards)" \
  "Corpus at $S3 . The 12M corpus used by published runs is untouched at .../pubchem_filtered/"
echo "PUBCHEM_DL_DONE rc=$rc"
exit $rc

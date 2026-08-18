#!/usr/bin/env bash
# Wait for the 124M tokenized corpus to be built, verify it is big enough to be honest, then
# start training. Runs detached on each training box so the whole pipeline is autonomous.
#
#   Usage: unsup124_chain.sh <manifest.json> <tag> <max_hours> <min_unique_molecules>
#
# The gate is the point: unsup_100M is only an honest name if the corpus really holds >=100M
# UNIQUE molecules. If dedup came up short this refuses to launch and alerts, rather than
# training 37h and mislabelling the result.
set -uo pipefail
cd /home/ec2-user/CLIMB

MANIFEST=${1:?manifest}
TAG=${2:?tag}
MAX_HOURS=${3:?max_hours}
MIN_UNIQUE=${4:?min unique molecules}

export CLIMB_WORKER=$TAG
PY=/home/ec2-user/venvs/climb/bin/python
REPORT_URI=s3://climb-s3-bucket/tokenized_sources/pubchem_124m_full_tokenized_pkl/_corpus_report.json
LOG=/home/ec2-user/unsup124_chain_${TAG}.log

say(){ echo "[chain-$TAG $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "waiting for corpus report at $REPORT_URI (need >= $MIN_UNIQUE unique)"
DEADLINE=$(( $(date +%s) + 8*3600 ))
while true; do
  if aws s3 cp "$REPORT_URI" /tmp/corpus_report_$TAG.json --only-show-errors 2>/dev/null; then
    ROWS=$($PY -c "import json;print(json.load(open('/tmp/corpus_report_$TAG.json')).get('rows_written',0))" 2>/dev/null || echo 0)
    if [ "${ROWS:-0}" -gt 0 ]; then
      say "corpus report present: rows_written=$ROWS"
      break
    fi
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "ABORT: corpus not ready after 8h"
    bash scripts/notify.sh ALERT "[$TAG] corpus never became ready - NOT launching" \
      "Waited 8h for $REPORT_URI. No training started. Box left up."
    exit 1
  fi
  sleep 120
done

UNIQ=$($PY -c "import json;print(json.load(open('/tmp/corpus_report_$TAG.json'))['unique_canonical_smiles'])")
say "corpus unique molecules: $UNIQ"
if [ "$UNIQ" -lt "$MIN_UNIQUE" ]; then
  say "ABORT: $UNIQ unique < required $MIN_UNIQUE"
  bash scripts/notify.sh ALERT "[$TAG] corpus too small - NOT launching" \
    "Corpus holds $UNIQ unique molecules, this worker needs >= $MIN_UNIQUE. Training NOT started so the run cannot be mislabelled. Box left up for a decision on the largest honest budget."
  exit 1
fi

say "gate passed - starting unsup124_run.sh"
exec bash scripts/unsup124_run.sh "$MANIFEST" "$TAG" "$MAX_HOURS"

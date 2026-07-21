#!/usr/bin/env bash
# Push locally-computed artifacts to S3 so they are not single-copy on one laptop.
#
# The scaffold 5-fold CV panels are produced locally (the box-side pass emits only the
# single-split eval), as is the Tanimoto novelty table behind Fig I1. Together that is several
# hours of compute across ~48 runs which, until now, existed nowhere else -- a lost or wiped
# laptop would mean recomputing all of it.
#
# Safe against the sync bug that reverted completed runs: this only ever ADDS
# moleculenet_cv/ (and a derived/ prefix) which no box holds locally, and `aws s3 sync` without
# --delete never removes destination-only files -- the same property that spared the encoders.
set -uo pipefail
cd "$(dirname "$0")/.."
S3=s3://climb-s3-bucket/experiments
n=0
for d in $(find figure_data -type d -name moleculenet_cv | sort); do
  rel=${d#figure_data/}                   # <wave>/<run>/moleculenet_cv
  wave=${rel%%/*}
  rest=${rel#*/}                          # <run>/moleculenet_cv
  run=${rest%%/*}
  aws s3 sync "$d" "$S3/$wave/$run/moleculenet_cv" --only-show-errors && n=$((n+1))
  echo "  [$n] $wave/$run"
done
echo "synced $n moleculenet_cv dirs"

# Derived analysis products that are not tied to a single run
if [ -d figure_data/_tanimoto ]; then
  aws s3 sync figure_data/_tanimoto s3://climb-s3-bucket/derived/tanimoto \
    --exclude "_cache/*" --only-show-errors
  echo "synced derived/tanimoto (corpus similarity for Fig I1)"
fi
echo "BACKUP_DONE"

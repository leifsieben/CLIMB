#!/usr/bin/env bash
# Pull the finished unsup124 runs from S3 into the LOCAL figure_data mirror the notebook reads.
#
# Training writes to s3://climb-s3-bucket/experiments/climb_v2_phase2/<run>/, but
# climb_figures.ipynb reads figure_data/climb_v2_phase2/<run>/. Without this step the runs exist
# but never appear in Figs A2.a/A2.b -- notebook_cells/12.py only lists the 50M/100M rungs once
# rows for them exist in the CV frame.
#
# Encoders are skipped by default: they are ~166 MB each and the figures only need the metrics
# and the two eval directories. Pass --with-encoder if you want them locally too.
set -euo pipefail
cd "$(dirname "$0")/.."

S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
DEST=figure_data/climb_v2_phase2
RUNS=${RUNS:-"unsup_50M unsup_100M unsup_8M_c124"}
EXCLUDE="--exclude */encoder/* --exclude */tokenizer/*"
[ "${1:-}" = "--with-encoder" ] && EXCLUDE="--exclude */tokenizer/*"

for r in $RUNS; do
  if ! aws s3 ls "$S3/$r/verified.json" >/dev/null 2>&1; then
    echo "SKIP $r - no verified.json in S3 (run not complete; refusing to pull a partial run)"
    continue
  fi
  echo "pulling $r"
  mkdir -p "$DEST/$r"
  # shellcheck disable=SC2086
  aws s3 sync "$S3/$r" "$DEST/$r" $EXCLUDE --only-show-errors
  for f in metrics.jsonl verified.json metadata.json run_status.json; do
    [ -f "$DEST/$r/$f" ] && echo "   $f ok"
  done
  for s in moleculenet moleculenet_cv; do
    n=$(wc -l < "$DEST/$r/$s/test_predictions.csv" 2>/dev/null || echo 0)
    echo "   $s/test_predictions.csv: $n lines"
  done
done
echo
echo "Now re-run the notebook. notebook_cells/12.py picks the 50M/100M rungs up automatically"
echo "once DF_CV has rows for them; unsup_8M_c124 is deliberately NOT matched by the"
echo "unsup_(\\d+M)\$ ladder regex, so it will not appear as a ladder rung."

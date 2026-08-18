#!/usr/bin/env bash
# P3: CBS end-to-end fine-tune of the best-two PRETRAINED CLIMB encoders (unsup_8M, skip_dense_8M).
# This is the SI-a 5/6 gap -- CBS is the only panel with no e2e run of a pretrained encoder.
#
# Note: two dirs named unsup_8M_e2e / skip_dense_8M_e2e already existed but were SMOKE stubs
# (n_folds=1, epochs=2, NEF1 0.0, AUC 0.516/0.332). They are quarantined to *_SMOKE_STUB, and
# cbs_e2e.py::done() now requires 5 folds + the full epoch budget + all seeds, so a stub can never
# again be mistaken for a result.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/cbs_e2e_run.log
say() { echo "[p3] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
[ -f data/cbs.csv ] || aws s3 cp s3://climb-s3-bucket/datasets/cbs.csv data/cbs.csv --only-show-errors
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ -f data/cbs.csv ] || { say "FATAL no cbs.csv -> staying UP"; exit 1; }

~/venvs/climb/bin/python scripts/cbs_e2e.py >> "$LOG" 2>&1
rc=$?
say "cbs_e2e rc=$rc"
if [ -f figure_data/CBS_E2E_DONE ]; then
  for r in unsup_8M_e2e skip_dense_8M_e2e; do
    [ -d "figure_data/cbs_benchmark/$r" ] && aws s3 cp --recursive "figure_data/cbs_benchmark/$r" \
      "s3://climb-s3-bucket/experiments/cbs_benchmark/$r" --only-show-errors
  done
  say "COMPLETE -> shutdown"; sudo shutdown -h now
else
  say "INCOMPLETE -> staying UP for inspection"
fi

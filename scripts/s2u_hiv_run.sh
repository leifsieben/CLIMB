#!/usr/bin/env bash
# ASK 9: the one missing cell in the 108-cell grid -- s2u_dense (sup->unsup, desc) has no HIV.
# Frozen-probe eval of HIV only, on the three checkpoints that already exist. No pretraining.
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
TOK=${TOK:-experiments/_tok_s2u}
LOG=analysis/s2u_hiv.log
mkdir -p analysis
say(){ echo "[s2uhiv $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors
RUNS="s2u_dense_from8M_s0 s2u_dense_from8M_s1 s2u_dense_from8M_s2"
ok=1
for r in $RUNS; do
  d=figure_data/climb_v2_phase2/$r
  mkdir -p "$d"
  aws s3 sync "$S3/$r/encoder" "$d/encoder" --only-show-errors
  # Stage the EXISTING results too. Merging presupposes a destination: on a fresh box the
  # destination summary does not exist, the merge keeps 0 rows, and the single-dataset file is
  # then synced back OVER the full one on S3. That is how the first attempt at this top-up
  # silently dropped BACE/QM7/Tox21 from all three dirs.
  aws s3 sync "$S3/$r/moleculenet_cv" "$d/moleculenet_cv" --only-show-errors
  if [ ! -s "$d/moleculenet_cv/moleculenet_summary.csv" ]; then
    say "$r: no existing summary staged -- refusing to write a single-dataset file over S3"
    ok=0; continue
  fi
  if [ ! -f "$d/encoder/model.safetensors" ] && [ ! -f "$d/encoder/pytorch_model.bin" ]; then
    say "$r: NO ENCODER WEIGHTS -> skipping"; ok=0; continue
  fi
  # HIV goes into moleculenet_cv/ alongside the arm's existing datasets, so it is written to a
  # temp dir and merged: eval_v2 opens moleculenet_summary.csv with "w" and would otherwise
  # delete every other dataset this arm already has.
  tmp=$(mktemp -d)
  $PY eval_v2.py --encoder "$d/encoder" --tokenizer "$TOK" --output_dir "$tmp" \
      --cv_folds 5 --head_seeds 0 1 2 --datasets HIV >> "$LOG" 2>&1
  if grep -q "^HIV,.*,fold0," "$tmp/moleculenet_summary.csv" 2>/dev/null; then
    $PY scripts/merge_summary_rows.py "$tmp/moleculenet_summary.csv" \
        "$d/moleculenet_cv/moleculenet_summary.csv" HIV >> "$LOG" 2>&1 \
      && say "$r: HIV merged" || { say "$r: MERGE FAILED"; ok=0; }
  else
    say "$r: HIV eval produced no fold rows"; ok=0
  fi
  rm -rf "$tmp"
  aws s3 sync "$d/moleculenet_cv" "$S3/$r/moleculenet_cv" --only-show-errors
done

if [ "$ok" = "1" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP"; fi

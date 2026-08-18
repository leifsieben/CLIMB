#!/usr/bin/env bash
# Laptop-independent job queue for the concat/FigF box. Nothing here depends on a local session.
#
# REWRITTEN 2026-08-18 after the first version shut the box down having produced NOTHING.
# Two independent faults, both now fixed:
#
#  1. NO INPUT STAGING. The box had never been given data/cbs.csv, the MoleculeACE CSVs, the
#     unsup_8M encoder, or the SMILES source. All three jobs died on FileNotFoundError within
#     seconds. Every input is now staged from S3 up front and the run aborts if staging fails.
#  2. COMPLETION WAS ASSERTED, NOT VERIFIED. The old script ran each step, ignored its exit
#     status, then unconditionally wrote QUEUE_DONE and powered off -- so three crashed jobs
#     reported as a clean finish. Completion is now derived from the OUTPUT FILES existing;
#     the box shuts down only if every one of them does, and otherwise stays up for inspection.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis analysis/rigor figure_data/_bench data
LOG=analysis/queue.log
say() { echo "[queue] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "supervisor start (v2, staged+gated)"

# ---- 0. stage every input this queue needs -------------------------------------------------
S3=s3://climb-s3-bucket
[ -f data/cbs.csv ] || aws s3 cp $S3/datasets/cbs.csv data/cbs.csv --only-show-errors
[ "$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)" -ge 30 ] || {
  mkdir -p chemeleon_suite/data/moleculeace
  aws s3 sync $S3/datasets/moleculeace/ chemeleon_suite/data/moleculeace/ --only-show-errors; }
[ -f figure_data/climb_v2_phase2/unsup_8M/encoder/model.safetensors ] || {
  mkdir -p figure_data/climb_v2_phase2/unsup_8M/encoder
  aws s3 sync $S3/experiments/climb_v2_phase2/unsup_8M/encoder \
    figure_data/climb_v2_phase2/unsup_8M/encoder --only-show-errors; }
[ -f figure_data/climb_v2_phase2/ecfp4_anchor/moleculenet/test_predictions.csv ] || {
  mkdir -p figure_data/climb_v2_phase2/ecfp4_anchor/moleculenet
  aws s3 cp $S3/experiments/climb_v2_phase2/ecfp4_anchor/moleculenet/test_predictions.csv \
    figure_data/climb_v2_phase2/ecfp4_anchor/moleculenet/test_predictions.csv --only-show-errors; }
[ -f figure_data/_tokenizer/tokenizer.json ] || {
  mkdir -p figure_data/_tokenizer; aws s3 sync $S3/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

stage_ok=1
for f in data/cbs.csv figure_data/climb_v2_phase2/unsup_8M/encoder/model.safetensors \
         figure_data/climb_v2_phase2/ecfp4_anchor/moleculenet/test_predictions.csv; do
  [ -f "$f" ] || { say "FATAL staging missing $f"; stage_ok=0; }
done
n=$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)
[ "$n" -ge 30 ] || { say "FATAL staging moleculeace=$n/30"; stage_ok=0; }
[ "$stage_ok" = "1" ] || { say "staging failed -> staying UP"; exit 1; }
say "inputs staged OK"

# ---- 1. SI fig c: CheMeleon featurization timing -------------------------------------------
BENCH=figure_data/_bench/featurization_timing_chemeleon.json
if [ ! -f "$BENCH" ]; then
  say "bench chemeleon"
  ~/venvs/chemeleon/bin/python scripts/bench_featurization.py \
    --bench_chemeleon --skip_rdkit --skip_encoder --devices cuda,cpu --cpu_threads 0,1 \
    --n_molecules 1000 --repeats 5 \
    --hardware_label "AWS g5.2xlarge (NVIDIA A10G 24GB, 8 vCPU)" \
    --out "$BENCH" >> "$LOG" 2>&1
  [ -f "$BENCH" ] && aws s3 cp "$BENCH" $S3/experiments/_bench/featurization_timing_chemeleon.json --only-show-errors
fi

# ---- 2. Fig F on the three non-MolNet panels, BOTH arms ------------------------------------
for emb in chemeleon climb; do
  PY=~/venvs/chemeleon/bin/python; [ "$emb" = "climb" ] && PY=~/venvs/climb/bin/python
  OUT=analysis/rigor/concat_panels_${emb}.csv
  if [ ! -f "$OUT" ]; then
    say "figF panels $emb"
    CONCAT_EMB=$emb CONCAT_PANELS="MoleculeACE CBS Ames" $PY scripts/concat_redundancy_panels.py >> "$LOG" 2>&1
    [ -f "$OUT" ] && aws s3 cp "$OUT" $S3/experiments/analysis_rigor/concat_panels_${emb}.csv --only-show-errors
  fi
done

# ---- 3. Ames scoring for the Fig F prediction dumps ----------------------------------------
if [ -x .venv_polaris/bin/python ]; then
  for emb in chemeleon climb; do
    d=figure_data/chemeleon_suite/polaris/concat_${emb}
    [ -f "$d/test_predictions.csv" ] && { .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py "$d" >> "$LOG" 2>&1
      aws s3 cp --recursive "$d" $S3/experiments/chemeleon_suite/polaris/concat_${emb} --only-show-errors; }
  done
fi

# ---- 4. completion is DERIVED from outputs, never from reaching this line -------------------
missing=""
for f in "$BENCH" analysis/rigor/concat_panels_chemeleon.csv analysis/rigor/concat_panels_climb.csv; do
  [ -s "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  say "INCOMPLETE, missing:$missing -> staying UP for inspection"
  exit 1
fi
say "ALL QUEUED JOBS VERIFIED COMPLETE"
touch figure_data/QUEUE_DONE
sudo shutdown -h now

#!/usr/bin/env bash
# Run the fig_B evaluation battery for one or more rungs.
#
# Pretraining being finished does NOT put a rung on fig_B. The figure reads the 5-fold scaffold CV
# tree (moleculenet_cv/), MoleculeACE, and the Polaris Ames scores -- and all four completed rungs
# have only the single-split moleculenet/, which is not comparable to any other arm and must never
# be pooled with one. skip_dense_50M_c124 has no eval at all: its run died before the eval stage.
#
# Usage: figB_eval_run.sh <run_id> [run_id ...]
#
# Each step is gated on ITS OWN artifact, never on a neighbour's: a rung that has MoleculeACE but
# no CV must still run the CV. Shutdown happens only when every requested rung has both.
set -u
set -o pipefail
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket
LOG=analysis/figB_eval.log
mkdir -p analysis figure_data
say () { echo "[eval] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/experiments/climb_v2_phase2/_eval_logs/$(hostname).log" --only-show-errors; exit 1; }

RUNS="$*"
[ -n "$RUNS" ] || abort "no run ids given"
say "battery for: $RUNS"

# The six panels, from figures/arms.py PANEL_ORDER: MoleculeACE, HIV, BACE, Ames, Tox21, QM7.
# BACE/Tox21/QM7/HIV come from the CV tree; Ames is Polaris; MoleculeACE is its own track.
CV_DATASETS="BACE Tox21 QM7 HIV"

[ -f figure_data/_tokenizer/tokenizer.json ] || {
  mkdir -p figure_data/_tokenizer
  aws s3 sync $S3/tokenizer_10M figure_data/_tokenizer --only-show-errors || abort "tokenizer sync failed"; }
# The task lists are .txt and the code-only deploy ships none: MoleculeACE failed a whole pass once
# for exactly this, with the 30 CSVs present and the list absent.
mkdir -p figure_data/chemeleon_suite/tasks
aws s3 sync $S3/experiments/chemeleon_suite/tasks figure_data/chemeleon_suite/tasks --only-show-errors || true
for t in moleculeace_tasks.txt polaris_tasks.txt; do
  [ -s "figure_data/chemeleon_suite/tasks/$t" ] || say "WARNING: task list $t missing or empty"
done

for run in $RUNS; do
  enc=experiments/climb_v2_phase2/$run/encoder
  mkdir -p "$enc"
  aws s3 sync "$S3/experiments/climb_v2_phase2/$run/encoder" "$enc" --only-show-errors || abort "encoder sync failed for $run"
  # Test for WEIGHTS: mkdir -p above guarantees the directory, so a directory test proves nothing.
  [ -f "$enc/model.safetensors" ] || abort "$run has no encoder weights -- cannot evaluate"
  say "$run: encoder staged"

  cv=figure_data/climb_v2_phase2/$run/moleculenet_cv
  if [ ! -f "$cv/moleculenet_summary.csv" ]; then
    say "$run: 5-fold scaffold CV over $CV_DATASETS"
    $PY eval_v2.py --encoder "$enc" --tokenizer figure_data/_tokenizer --output_dir "$cv" \
        --head mlp --head_seeds 0 1 2 --pool mean --standardize zscore --max_length 256 \
        --datasets $CV_DATASETS --cv_folds 5 2>&1 | tail -20 | tee -a "$LOG"
    [ -f "$cv/moleculenet_summary.csv" ] || abort "$run: CV produced no moleculenet_summary.csv"
    aws s3 cp --recursive "$cv" "$S3/experiments/climb_v2_phase2/$run/moleculenet_cv" --only-show-errors \
      || abort "$run: CV upload failed"
    say "$run: CV done and uploaded"
  else
    say "$run: CV already present -- skipping"
  fi

  for track in moleculeace polaris; do
    out=figure_data/chemeleon_suite/$track/$run
    if [ ! -f "$out/results.csv" ]; then
      say "$run: $track"
      $PY scripts/chemeleon_suite_run.py --track $track --featurizer encoder --model "$run" \
          --encoder "$enc" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 \
          2>&1 | tail -15 | tee -a "$LOG"
      [ -f "$out/results.csv" ] || abort "$run: $track produced no results.csv"
      aws s3 cp --recursive "$out" "$S3/experiments/chemeleon_suite/$track/$run" --only-show-errors \
        || abort "$run: $track upload failed"
      say "$run: $track done and uploaded"
    else
      say "$run: $track already present -- skipping"
    fi
  done
done

# ---- completion is per-artifact, across every requested rung -------------------------------------
missing=0
for run in $RUNS; do
  for f in "figure_data/climb_v2_phase2/$run/moleculenet_cv/moleculenet_summary.csv" \
           "figure_data/chemeleon_suite/moleculeace/$run/results.csv" \
           "figure_data/chemeleon_suite/polaris/$run/results.csv"; do
    [ -f "$f" ] || { say "MISSING $f"; missing=$((missing+1)); }
  done
done
aws s3 cp "$LOG" "$S3/experiments/climb_v2_phase2/_eval_logs/$(hostname).log" --only-show-errors
if [ "$missing" -eq 0 ]; then
  say "ALL ARTIFACTS PRESENT for: $RUNS"
  say "NOTE: Polaris Ames still needs scripts/chemeleon_suite_score_polaris.py run OFF-BOX to write polaris_scores.csv"
  [ "${EVAL_SHUTDOWN:-0}" = "1" ] && { say "EVAL_SHUTDOWN=1 -- terminating"; sudo shutdown -h now; }
  say "EVAL_SHUTDOWN unset -- staying up"
else
  abort "$missing artifact(s) missing"
fi

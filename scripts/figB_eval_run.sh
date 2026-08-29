#!/usr/bin/env bash
# Run the fig_B evaluation battery for one or more rungs.
#
# Pretraining being finished does NOT put a rung on fig_B. The figure reads the 5-fold scaffold CV
# tree (moleculenet_cv/), MoleculeACE, and the Polaris Ames predictions -- and all four completed
# rungs have only the single-split moleculenet/, which is not comparable to any other arm and must
# never be pooled with one. skip_dense_50M_c124 has no eval at all: its run died before the eval
# stage.
#
# Usage: figB_eval_run.sh <run_id> [run_id ...]
#
# Each step is gated on ITS OWN artifact, never on a neighbour's: a rung that has MoleculeACE but
# no CV must still run the CV. Shutdown happens only when every requested rung has all of them.
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

# ---- the evaluation environment must be the one the ladder was measured in ----------------------
# skip_dense_48M's 208-descriptor defect was rdkit-pypi shadowing rdkit, and the SAME shadowing is
# live on these AMIs: `rdkit 2025.9.2` and `rdkit-pypi 2022.9.5` are both installed and `import
# rdkit` resolves to 2022.09.5. That is not a cosmetic version difference. 2022.09.5 canonicalizes
# organometallics differently (C[Hg]Cl vs [CH3][Hg][Cl]), so DeepChem's Raw featurizer keeps 7,831
# Tox21 molecules where the reference environment keeps 7,823 -- a 77,946-row dump instead of
# 77,864. rescore_tox21.py refuses anything but 77,864, so the whole Tox21 panel is lost, and the
# scaffold folds every panel is scored on are cut from these same canonical SMILES.
#
# Repair first, then test the BEHAVIOUR rather than the version string: a version string can agree
# while the import resolves somewhere else entirely, which is exactly how this hid.
# The two dists share the SAME `rdkit` package directory, so uninstalling rdkit-pypi deletes files
# the surviving rdkit also owns and leaves a half-populated package -- `rdkit.Chem` imports but has
# no MolToSmiles. Always reinstall after the purge; never uninstall alone.
if $PY -m pip show rdkit-pypi >/dev/null 2>&1; then
  say "rdkit-pypi is shadowing rdkit -- purging and reinstalling the pinned rdkit"
  $PY -m pip uninstall -y rdkit-pypi 2>&1 | tail -2 | tee -a "$LOG"
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" 2>&1 | tail -2 | tee -a "$LOG"
fi
$PY - <<'PYCHK' 2>&1 | tee -a "$LOG"
import sys
import rdkit
from rdkit import Chem
canon = Chem.MolToSmiles(Chem.MolFromSmiles("C[Hg]Cl"))
print(f"[eval] rdkit {rdkit.__version__} canonicalizes C[Hg]Cl as {canon}")
if canon != "[CH3][Hg][Cl]":
    sys.exit("rdkit canonicalization is not the reference environment's")
PYCHK
[ "${PIPESTATUS[0]}" = "0" ] || abort "rdkit is not the reference environment -- Tox21 would be scored on 7,831 molecules against a ladder built on 7,823"

# The task lists live in the REPO, at the path chemeleon_suite_run.py actually reads
# (ROOT/chemeleon_suite/tasks). An earlier version of this script checked figure_data/... instead
# and warned on every box about files that were present all along -- a check against the wrong path
# is worse than no check, because it trains you to ignore the warning that matters.
for t in moleculeace_tasks.txt polaris_tasks.txt; do
  [ -s "chemeleon_suite/tasks/$t" ] || abort "task list chemeleon_suite/tasks/$t missing or empty"
done
# fig_B's Ames panel is tdcommons/ames alone. The full list is 28 tasks x 3 seeds of work that no
# panel reads, and every extra task is another chance to abort the stage that does matter.
printf 'tdcommons/ames\n' > chemeleon_suite/tasks/polaris_tasks.txt
say "task lists: $(wc -l < chemeleon_suite/tasks/moleculeace_tasks.txt) moleculeace, polaris trimmed to tdcommons/ames"

# The per-molecule predictions are NOT a bonus artifact. moleculenet_cv_tox21fixed/ and
# _qm7clamped/ are rescored from them off-box, and the aggregator picks one subdir per panel and
# DROPS any arm missing it -- so a rung that ships a summary without predictions is not merely less
# precise on Tox21 and QM7, it silently vanishes from those two panels. eval_v2.py dumps them under
# a try/except that only warns, so the summary can appear without them. Check for them explicitly.
have_preds () {  # <file> <dataset...>: file exists and names every dataset
  local f=$1; shift
  [ -s "$f" ] || return 1
  local d
  for d in "$@"; do grep -q "^$d," "$f" || grep -q ",$d," "$f" || return 1; done
  return 0
}

for run in $RUNS; do
  enc=experiments/climb_v2_phase2/$run/encoder
  mkdir -p "$enc"
  aws s3 sync "$S3/experiments/climb_v2_phase2/$run/encoder" "$enc" --only-show-errors || abort "encoder sync failed for $run"
  # Test for WEIGHTS: mkdir -p above guarantees the directory, so a directory test proves nothing.
  [ -f "$enc/model.safetensors" ] || abort "$run has no encoder weights -- cannot evaluate"
  say "$run: encoder staged"

  cv=figure_data/climb_v2_phase2/$run/moleculenet_cv
  if [ ! -f "$cv/moleculenet_summary.csv" ] || ! have_preds "$cv/test_predictions.csv" $CV_DATASETS; then
    say "$run: 5-fold scaffold CV over $CV_DATASETS"
    $PY eval_v2.py --encoder "$enc" --tokenizer figure_data/_tokenizer --output_dir "$cv" \
        --head mlp --head_seeds 0 1 2 --pool mean --standardize zscore --max_length 256 \
        --datasets $CV_DATASETS --cv_folds 5 2>&1 | tail -20 | tee -a "$LOG"
    [ -f "$cv/moleculenet_summary.csv" ] || abort "$run: CV produced no moleculenet_summary.csv"
    have_preds "$cv/test_predictions.csv" $CV_DATASETS \
      || abort "$run: CV summary exists but test_predictions.csv is missing or does not cover $CV_DATASETS -- Tox21/QM7 cannot be rescored off-box and this rung would be dropped from those panels"
    # rescore_tox21.py accepts EXACTLY this count (93,876 cells - 16,012 masked). Anything else and
    # it writes nothing, so the rung silently falls back to the stale protocol on the Tox21 panel.
    n_tox=$(awk -F, 'NR>1 && $1=="Tox21"' "$cv/test_predictions.csv" | wc -l | tr -d ' ')
    [ "$n_tox" = "77864" ] \
      || abort "$run: Tox21 dump has $n_tox rows, need 77864 -- the molecule set is not the ladder's and rescore_tox21.py will refuse it"
    say "$run: Tox21 dump is $n_tox rows -- matches the ladder"
    aws s3 cp --recursive "$cv" "$S3/experiments/climb_v2_phase2/$run/moleculenet_cv" --only-show-errors \
      || abort "$run: CV upload failed"
    say "$run: CV done and uploaded"
  else
    say "$run: CV already present with predictions -- skipping"
  fi

  # MoleculeACE is scored on the box (results.csv carries the subset=overall / metric=rmse rows the
  # panel means over). Polaris is NOT: it is scored off-box against held-out labels, so its
  # results.csv is header-only BY DESIGN and test_predictions.csv is the real deliverable.
  for track in moleculeace polaris; do
    out=figure_data/chemeleon_suite/$track/$run
    case $track in
      moleculeace) want="$out/results.csv" ;;
      polaris)     want="$out/test_predictions.csv" ;;
    esac
    if [ ! -s "$want" ]; then
      say "$run: $track"
      $PY scripts/chemeleon_suite_run.py --track $track --featurizer encoder --model "$run" \
          --encoder "$enc" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 \
          2>&1 | tail -15 | tee -a "$LOG"
      [ -s "$want" ] || abort "$run: $track produced no $(basename "$want")"
      if [ "$track" = moleculeace ]; then
        grep -q ',overall,rmse,' "$out/results.csv" \
          || abort "$run: moleculeace results.csv has no subset=overall,metric=rmse rows -- the panel value cannot be computed from it"
      fi
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
  cv=figure_data/climb_v2_phase2/$run/moleculenet_cv
  have_preds "$cv/test_predictions.csv" $CV_DATASETS || { say "MISSING/INCOMPLETE $cv/test_predictions.csv"; missing=$((missing+1)); }
  for f in "$cv/moleculenet_summary.csv" \
           "figure_data/chemeleon_suite/moleculeace/$run/results.csv" \
           "figure_data/chemeleon_suite/polaris/$run/test_predictions.csv"; do
    [ -s "$f" ] || { say "MISSING $f"; missing=$((missing+1)); }
  done
done
if [ "$missing" -eq 0 ]; then
  say "ALL ARTIFACTS PRESENT for: $RUNS"
  aws s3 cp "$LOG" "$S3/experiments/climb_v2_phase2/_eval_logs/$(hostname).log" --only-show-errors
  say "NOTE: Polaris Ames still needs scripts/chemeleon_suite_score_polaris.py run OFF-BOX to write polaris_scores.csv"
  [ "${EVAL_SHUTDOWN:-0}" = "1" ] && { say "EVAL_SHUTDOWN=1 -- shutting down"; sudo shutdown -h now; }
  say "EVAL_SHUTDOWN unset -- staying up"
else
  abort "$missing artifact(s) missing"
fi

#!/usr/bin/env bash
# CheMeleon-frozen model-seed replicates for fig A1/A2 (BACE, CBS, Tox21).
#
# The arm had ONE run behind those three bars, so its error bar was fold spread only. Two more
# replicates on disjoint head-seed triples ({3,4,5}, {6,7,8}) turn that into a real model-seed SD,
# matching how every CLIMB arm is estimated. QM7 is NOT re-run: _s1/_s2 already carry it.
#
# THE THING THAT MAKES THIS BOX DIFFERENT FROM THE LAST ONES.
# Tox21's published value depends on the environment that parses it. The reference environment
# (the laptop, and every run currently in fig_A) yields 77,864 masked prediction rows; the boxes
# we ran in August parse a few molecules differently and land ~0.008 off, which is exactly the
# drift that cost us the Tox21 column in fig_C2/fig_D. So this box PINS the parsing stack to the
# reference versions and REFUSES to keep Tox21 output that does not reproduce 77,864 rows.
# CheMeleon needs chemprop>=2.2 (Python >=3.11), which is why this cannot run on the laptop's
# 3.9 reference venv -- pinning the parsing stack is what makes the two comparable anyway.
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments
REF_ROWS=77864
LOG=analysis/chemeleon_replicates.log
mkdir -p analysis
say(){ echo "[cfrep $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

PY=$HOME/venvs/cf/bin/python
if [ ! -x "$PY" ]; then
  say "building pinned venv"
  { command -v python3.12 >/dev/null && python3.12 -m venv "$HOME/venvs/cf"; } \
    || { command -v python3.11 >/dev/null && python3.11 -m venv "$HOME/venvs/cf"; } \
    || { say "FATAL no python3.11/3.12 -> staying UP"; exit 1; }
  $HOME/venvs/cf/bin/pip -q install --upgrade pip
  # versions read off the reference environment: rdkit 2025.09.2, numpy 2.0.2,
  # scikit-learn 1.6.1, deepchem 2.8.0 -- these decide how Tox21 parses.
  $HOME/venvs/cf/bin/pip -q install "numpy==2.0.2" "rdkit==2025.9.2" "scikit-learn==1.6.1" \
      "deepchem==2.8.0" "chemprop>=2.2.0" torch xgboost pandas || {
      say "FATAL pinned install failed -> staying UP"; exit 1; }
fi

$PY -c "import rdkit,numpy,sklearn,deepchem,chemprop
print('rdkit',rdkit.__version__,'numpy',numpy.__version__,'sklearn',sklearn.__version__,
      'deepchem',deepchem.__version__,'chemprop',chemprop.__version__)" 2>&1 | tee -a "$LOG" \
  || { say "FATAL pinned imports broken -> staying UP"; exit 1; }

[ -f data/cbs.csv ] || aws s3 cp s3://climb-s3-bucket/datasets/cbs.csv data/cbs.csv --only-show-errors
mkdir -p figure_data/climb_v2_phase2 figure_data/cbs_benchmark
for suf in _s1 _s2; do
  aws s3 sync "$S3/climb_v2_phase2/chemeleon_frozen$suf" \
      "figure_data/climb_v2_phase2/chemeleon_frozen$suf" --only-show-errors
done
aws s3 sync "$S3/climb_v2_phase2/chemeleon_frozen" \
    figure_data/climb_v2_phase2/chemeleon_frozen --only-show-errors

$PY scripts/anchor_seed_replicates.py chemeleon_frozen >> "$LOG" 2>&1
rc=$?
say "replicate driver rc=$rc"

# ---- gate: Tox21 must reproduce the reference row count, or its output is quarantined --------
bad=0
for suf in _s1 _s2; do
  d="figure_data/climb_v2_phase2/chemeleon_frozen$suf/moleculenet_cv_tox21fixed"
  f="$d/test_predictions.csv"
  n=$( [ -f "$f" ] && awk -F, '$1=="Tox21"' "$f" | wc -l | tr -d ' ' || echo 0 )
  if [ "$n" = "$REF_ROWS" ]; then
    say "Tox21$suf OK ($n rows == reference)"
  else
    say "Tox21$suf DRIFT ($n rows != $REF_ROWS) -> quarantining, NOT shipping"
    [ -d "$d" ] && mv "$d" "${d}.DRIFT_${n}rows"
    bad=1
  fi
done

# ---- upload whatever is valid, then decide on shutdown ---------------------------------------
for suf in _s1 _s2; do
  aws s3 sync "figure_data/climb_v2_phase2/chemeleon_frozen$suf" \
      "$S3/climb_v2_phase2/chemeleon_frozen$suf" --only-show-errors
  aws s3 sync "figure_data/cbs_benchmark/chemeleon_frozen$suf" \
      "$S3/cbs_benchmark/chemeleon_frozen$suf" --only-show-errors
done
say "uploaded"

# Completion is achieved work, never "a file appeared": BACE and CBS must carry fold rows in both
# replicate dirs. Tox21 drift alone does not block shutdown -- it is quarantined and reported.
ok=1
for suf in _s1 _s2; do
  grep -q "^BACE,.*,roc_auc,fold0," "figure_data/climb_v2_phase2/chemeleon_frozen$suf/moleculenet_cv/moleculenet_summary.csv" 2>/dev/null || ok=0
  grep -q "^cbs,.*,nef1,fold0," "figure_data/cbs_benchmark/chemeleon_frozen$suf/moleculenet_cv/moleculenet_summary.csv" 2>/dev/null || ok=0
done
if [ "$ok" = "1" ]; then
  say "COMPLETE (tox21_drift=$bad) -> shutdown"
  sudo shutdown -h now
else
  say "INCOMPLETE -> staying UP for inspection"
fi

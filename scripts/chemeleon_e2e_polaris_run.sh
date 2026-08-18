#!/usr/bin/env bash
# ASK 6: full 28-task Polaris for chemeleon_e2e, taking it from 39/66 all-suites coverage to 66/66
# so it clears fig_A1's >=60/66 admission floor. Until it does, fig_A panel (a) shows CheMeleon
# FROZEN while panel (b) shows CheMeleon E2E -- two different models under one comparator name.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/chemeleon_e2e_polaris.log
say() { echo "[cee] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
mkdir -p chemeleon_suite/data/polaris
aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors
n=$(grep -c . chemeleon_suite/tasks/polaris_tasks.txt 2>/dev/null || echo 0)
[ "$n" -ge 28 ] || { say "FATAL polaris_tasks.txt has $n tasks (want 28) -> staying UP"; exit 1; }

PY=~/venvs/chemeleon/bin/python
if ! $PY -c "import chemprop" 2>/dev/null; then
  say "building chemeleon venv"
  python3.12 -m venv ~/venvs/chemeleon
  $PY -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  $PY -m pip install -q "chemprop==2.3.1" rdkit deepchem==2.5.0 xgboost >> "$LOG" 2>&1
  [ -f scripts/molnet_box_bootstrap.sh ] && bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi
$PY -c "import chemprop" 2>/dev/null || { say "FATAL chemeleon venv -> staying UP"; exit 1; }

CHEM_ONLY=polaris_all $PY scripts/chemeleon_e2e_gaps.py >> "$LOG" 2>&1
say "predict rc=$?"

D=figure_data/chemeleon_suite/polaris/chemeleon_e2e
if [ -x .venv_polaris/bin/python ] || { python3.12 -m venv .venv_polaris >/dev/null 2>&1
     .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
     .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1; }; then
  .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py "$D" >> "$LOG" 2>&1
fi
t=$(tail -n +2 "$D/polaris_scores.csv" 2>/dev/null | cut -d, -f1 | sort -u | wc -l)
say "scored tasks=$t"
if [ "${t:-0}" -ge 20 ]; then
  aws s3 cp --recursive "$D" s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/chemeleon_e2e --only-show-errors
  say "COMPLETE ($t tasks) -> shutdown"; sudo shutdown -h now
else
  say "INCOMPLETE ($t tasks) -> staying UP"
fi

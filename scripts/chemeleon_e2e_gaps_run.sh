#!/usr/bin/env bash
# CheMeleon e2e gap-fill: MoleculeACE (30 targets) + hERG (Polaris). Builds both venvs it needs
# (chemprop for training, polaris for scoring hERG's withheld labels), runs, syncs, gated shutdown.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis
LOG=analysis/chemeleon_e2e_gaps.log
echo "[wrapper] start $(date -u +%FT%TZ)" >> "$LOG"

# --- chemprop venv (trains the D-MPNN from the CheMeleon foundation) ---
if [ ! -x ~/venvs/chemeleon/bin/chemprop ]; then
  echo "[wrapper] building chemeleon venv" >> "$LOG"
  python3.12 -m venv ~/venvs/chemeleon
  ~/venvs/chemeleon/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  ~/venvs/chemeleon/bin/python -m pip install -q "chemprop==2.3.1" >> "$LOG" 2>&1
fi
~/venvs/chemeleon/bin/chemprop --help >/dev/null 2>&1 || { echo "[wrapper] FATAL chemprop missing" >> "$LOG"; exit 1; }

# --- polaris venv (scores hERG; Polaris withholds test labels) ---
if [ ! -x .venv_polaris/bin/python ]; then
  echo "[wrapper] building polaris venv" >> "$LOG"
  python3.12 -m venv .venv_polaris
  .venv_polaris/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
  .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
fi

# --- run both cells ---
~/venvs/chemeleon/bin/python scripts/chemeleon_e2e_gaps.py >> "$LOG" 2>&1
echo "[wrapper] driver rc=$? $(date -u +%FT%TZ)" >> "$LOG"

# --- score hERG through Polaris, then sync ---
POL=figure_data/chemeleon_suite/polaris/chemeleon_e2e
if [ -f "$POL/test_predictions.csv" ]; then
  .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py "$POL" >> "$LOG" 2>&1
  aws s3 cp --recursive "$POL" s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/chemeleon_e2e --only-show-errors
fi

MACE=figure_data/chemeleon_suite/moleculeace/chemeleon_e2e
ok=0
[ -f "$MACE/verified.json" ] && ok=$((ok+1))
[ -f "$POL/polaris_scores.csv" ] && ok=$((ok+1))
echo "[wrapper] complete=$ok/2 $(date -u +%FT%TZ)" >> "$LOG"
if [ "$ok" -eq 2 ]; then
  touch figure_data/CHEMELEON_E2E_GAPS_DONE
  echo "[wrapper] done -> shutdown $(date -u +%FT%TZ)" >> "$LOG"; sudo shutdown -h now
else
  echo "[wrapper] incomplete -> staying UP for inspection $(date -u +%FT%TZ)" >> "$LOG"
fi

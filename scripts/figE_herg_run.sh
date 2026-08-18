#!/usr/bin/env bash
# (a) Fig-E arms on canonical panels + (b) hERG on mainline pretraining-seed replicates.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/figE_herg.log
echo "[wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
# polaris venv for hERG scoring (labels withheld). No pip upgrade: it corrupts pip on py3.12.
if [ ! -x .venv_polaris/bin/python ]; then
  python3.12 -m venv .venv_polaris
  .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
  .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
fi
.venv_polaris/bin/python -c "import polaris" 2>/dev/null || { echo "[wrapper] FATAL polaris venv" >> "$LOG"; exit 1; }
echo "tdcommons/herg" > chemeleon_suite/tasks/polaris_tasks.txt   # herg-only; driver refuses full dirs
~/venvs/climb/bin/python scripts/figE_and_herg_seeds.py >> "$LOG" 2>&1
echo "[wrapper] rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/FIGE_HERG_DONE ]; then
  echo "[wrapper] done -> shutdown" >> "$LOG"; sudo shutdown -h now
else
  echo "[wrapper] incomplete -> staying UP" >> "$LOG"
fi

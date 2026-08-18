#!/usr/bin/env bash
# (a) Fig-E arms on canonical panels + (b) hERG on mainline pretraining-seed replicates.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/figE_herg.log
echo "[wrapper] start $(date -u +%FT%TZ)" >> "$LOG"
# polaris venv. The AMI's python3.12 ensurepip can land a pip whose internals are broken
# (ImportError: open_rich_spinner) -- recreating the venv clears it; never `pip install --upgrade pip`
# here, that is what corrupted it the first time.
build_polaris_venv() {
  for attempt in 1 2 3; do
    [ -x .venv_polaris/bin/python ] && .venv_polaris/bin/python -c "import polaris" 2>/dev/null && return 0
    rm -rf .venv_polaris
    python3.12 -m venv .venv_polaris
    .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
    .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  done
  .venv_polaris/bin/python -c "import polaris" 2>/dev/null
}
build_polaris_venv || { echo "[wrapper] FATAL polaris venv after 3 attempts" >> "$LOG"; exit 1; }
~/venvs/climb/bin/python scripts/figE_and_herg_seeds.py >> "$LOG" 2>&1
echo "[wrapper] rc=$? $(date -u +%FT%TZ)" >> "$LOG"
if [ -f figure_data/FIGE_HERG_DONE ]; then
  echo "[wrapper] done -> shutdown" >> "$LOG"; sudo shutdown -h now
else
  echo "[wrapper] incomplete -> staying UP" >> "$LOG"
fi

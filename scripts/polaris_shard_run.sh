#!/usr/bin/env bash
# Full 28-task Polaris on a shard of the non-mainline encoders. SHARD/NSHARD via env.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/polaris_shard.log
echo "[shard $SHARD/$NSHARD] start $(date -u +%FT%TZ)" >> "$LOG"
build_polaris_venv() {
  for a in 1 2 3; do
    [ -x .venv_polaris/bin/python ] && .venv_polaris/bin/python -c "import polaris" 2>/dev/null && return 0
    rm -rf .venv_polaris; python3.12 -m venv .venv_polaris
    .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
    .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  done
  .venv_polaris/bin/python -c "import polaris" 2>/dev/null
}
build_polaris_venv || { echo "[shard] FATAL polaris venv" >> "$LOG"; exit 1; }
SHARD=$SHARD NSHARD=$NSHARD ~/venvs/climb/bin/python scripts/six_panel_herg.py >> "$LOG" 2>&1
rc=$?
echo "[shard] rc=$rc $(date -u +%FT%TZ)" >> "$LOG"
n=$(grep -c "^\[herg\] DONE" "$LOG")
if [ "$rc" -eq 0 ]; then echo "[shard] complete -> shutdown" >> "$LOG"; sudo shutdown -h now
else echo "[shard] incomplete -> staying UP" >> "$LOG"; fi

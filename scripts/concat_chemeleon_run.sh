#!/usr/bin/env bash
# Fig F second arm: CheMeleon redundancy/concat test (same head/splits/seeds as the CLIMB arm),
# plus P5 (unsup_48M MoleculeACE, the one hole in the Fig-B scaling ladder).
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/concat_chemeleon.log
echo "[concat] start $(date -u +%FT%TZ)" >> "$LOG"
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer; aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

# chemprop venv supplies the CheMeleon fingerprint featurizer
if [ ! -x ~/venvs/chemeleon/bin/python ]; then
  python3.12 -m venv ~/venvs/chemeleon
  ~/venvs/chemeleon/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  ~/venvs/chemeleon/bin/python -m pip install -q "chemprop==2.3.1" xgboost rdkit deepchem==2.5.0 >> "$LOG" 2>&1
  bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi

CONCAT_EMB=chemeleon ~/venvs/chemeleon/bin/python scripts/concat_redundancy.py >> "$LOG" 2>&1
echo "[concat] chemeleon rc=$? $(date -u +%FT%TZ)" >> "$LOG"
aws s3 cp analysis/rigor/concat_redundancy_chemeleon.csv \
  s3://climb-s3-bucket/experiments/analysis_rigor/concat_redundancy_chemeleon.csv --only-show-errors

# --- P5: unsup_48M MoleculeACE (one frozen probe, fills the Fig-B ladder hole) ---
P=unsup_48M
if [ ! -f figure_data/chemeleon_suite/moleculeace/$P/verified.json ]; then
  d=figure_data/_stage_p5/$P/encoder
  [ -f $d/model.safetensors ] || { mkdir -p $d; aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2/$P/encoder $d --only-show-errors; }
  ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
    --model $P --encoder $d --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
  aws s3 cp --recursive figure_data/chemeleon_suite/moleculeace/$P \
    s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$P --only-show-errors
fi

ok=0
[ -f analysis/rigor/concat_redundancy_chemeleon.csv ] && ok=$((ok+1))
[ -f figure_data/chemeleon_suite/moleculeace/$P/verified.json ] && ok=$((ok+1))
echo "[concat] complete=$ok/2 $(date -u +%FT%TZ)" >> "$LOG"
if [ "$ok" -eq 2 ]; then touch figure_data/CONCAT_CHEMELEON_DONE; sudo shutdown -h now
else echo "[concat] incomplete -> staying UP" >> "$LOG"; fi

#!/usr/bin/env bash
# Fig F's missing Ames panel for the CLIMB arm. It never ran because the Ames block reads
# chemeleon_suite/data/polaris/, which was not on S3 -- the same staging gap that broke the
# MoleculeACE and CBS steps. That data is on S3 now, so stage it and run Ames alone.
# BACE/Tox21/QM7 are NOT re-run: concat_redundancy.csv already has them, natively and on the same
# _scaffold_kfold_indices(...,0) splits as the canonical panels.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis analysis/rigor
LOG=analysis/figf_ames.log
say() { echo "[figf] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
mkdir -p chemeleon_suite/data/polaris
aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors
[ -f chemeleon_suite/data/polaris/tdcommons__ames.csv ] || { say "FATAL no ames csv -> staying UP"; exit 1; }
[ -f figure_data/climb_v2_phase2/unsup_8M/encoder/model.safetensors ] || {
  mkdir -p figure_data/climb_v2_phase2/unsup_8M/encoder
  aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2/unsup_8M/encoder \
    figure_data/climb_v2_phase2/unsup_8M/encoder --only-show-errors; }
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

# ALL THREE panels, not Ames alone: concat_redundancy_panels.py ends with
# pd.DataFrame(rows).to_csv(OUTFILE) -- a full REWRITE, not an append. Running only Ames would
# produce a file containing only Ames and silently drop the MoleculeACE and CBS rows that are
# already there. Re-running all three costs a little more and guarantees one consistent file.
say "running MoleculeACE + CBS + Ames (full rewrite, so all three must run together)"
CONCAT_EMB=climb CONCAT_PANELS="MoleculeACE CBS Ames" ~/venvs/climb/bin/python scripts/concat_redundancy_panels.py >> "$LOG" 2>&1

D=figure_data/chemeleon_suite/polaris/concat_climb
if [ -f "$D/test_predictions.csv" ]; then
  if [ ! -x .venv_polaris/bin/python ]; then
    python3.12 -m venv .venv_polaris
    .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
    .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  fi
  .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py "$D" >> "$LOG" 2>&1
  aws s3 cp --recursive "$D" s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/concat_climb --only-show-errors
fi
[ -f analysis/rigor/concat_panels_climb.csv ] && aws s3 cp analysis/rigor/concat_panels_climb.csv \
  s3://climb-s3-bucket/experiments/analysis_rigor/concat_panels_climb.csv --only-show-errors

# gate on ALL THREE panels being present, not just Ames
n=0
for t in MoleculeACE CBS Ames; do grep -q "^${t}," analysis/rigor/concat_panels_climb.csv 2>/dev/null && n=$((n+1)); done
if [ "$n" -eq 3 ]; then say "COMPLETE (3/3 panels) -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE ($n/3 panels) -> staying UP"; fi

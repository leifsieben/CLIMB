#!/usr/bin/env bash
# Finish the ONE missing panel on the corrupt_mtr box: the Polaris track.
#
# The run reported panels=2/3 and correctly stayed up. Cause was not the model or the data --
# .venv_polaris had a broken numpy ("No module named numpy._core._multiarray_umath"), the known
# venv corruption on this AMI. The venv existed and imported polaris, so the original build guard
# passed; this rebuilds unconditionally and verifies numpy AND polaris both import before use.
#
# Gated: uploads and powers off only if polaris_scores.csv really covers >=20 tasks.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
RUN_ID="${RUN_ID:?set RUN_ID}"
LOG=analysis/cmtr_polaris_fix.log
say() { echo "[fix] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start $RUN_ID"

rm -rf .venv_polaris
for a in 1 2 3; do
  python3.12 -m venv .venv_polaris
  .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
  .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  if .venv_polaris/bin/python -c "import numpy, polaris" 2>/dev/null; then say "venv OK (attempt $a)"; break; fi
  say "venv attempt $a failed"; rm -rf .venv_polaris
done
.venv_polaris/bin/python -c "import numpy, polaris" 2>/dev/null || { say "FATAL venv unbuildable -> staying UP"; exit 1; }

ENC=figure_data/_stage_cmtr/$RUN_ID/encoder
if [ ! -f "$ENC/model.safetensors" ]; then
  mkdir -p "$ENC"
  aws s3 sync s3://climb-s3-bucket/experiments/climb_v2_phase2/$RUN_ID/encoder "$ENC" --only-show-errors
fi
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

say "predict polaris"
~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track polaris --featurizer encoder \
  --model "$RUN_ID" --encoder "$ENC" --tokenizer figure_data/_tokenizer \
  --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
say "score polaris"
.venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py \
  "figure_data/chemeleon_suite/polaris/$RUN_ID" >> "$LOG" 2>&1

F=figure_data/chemeleon_suite/polaris/$RUN_ID/polaris_scores.csv
n=$(tail -n +2 "$F" 2>/dev/null | cut -d, -f1 | sort -u | wc -l)
if [ "${n:-0}" -ge 20 ]; then
  aws s3 cp --recursive "figure_data/chemeleon_suite/polaris/$RUN_ID" \
    "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/$RUN_ID" --only-show-errors
  say "COMPLETE ($n tasks) -> shutdown"
  sudo shutdown -h now
else
  say "INCOMPLETE ($n tasks) -> staying UP"
fi

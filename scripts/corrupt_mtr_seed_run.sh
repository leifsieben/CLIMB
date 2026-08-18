#!/usr/bin/env bash
# (c) One corrupt_mtr_8M pretraining replicate + its canonical-panel evals. RUN_ID via env.
# Two seeds run on two boxes in PARALLEL (user wants results ASAP), ~3h pretrain + ~1h evals each.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis configs
: "${RUN_ID:?set RUN_ID}"
LOG=analysis/${RUN_ID}.log
echo "[cmtr] start $RUN_ID $(date -u +%FT%TZ)" >> "$LOG"

[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer; aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ -f configs/descriptor_stats.json ] || aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors

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

# --- 1. pretrain (8M FP, shuffled MTR targets). _backup_to_s3 now ships encoder/ too. ---
~/venvs/climb/bin/python scripts/launch_v2_wave.py \
  --manifest experiments/climb_v2_phase2/manifests/corrupt_mtr_seeds.json \
  --run_id "$RUN_ID" --worker_name "box_$RUN_ID" >> "$LOG" 2>&1
echo "[cmtr] pretrain rc=$? $(date -u +%FT%TZ)" >> "$LOG"

ENC=experiments/climb_v2_phase2/$RUN_ID/encoder
if [ ! -f "$ENC/model.safetensors" ]; then echo "[cmtr] FATAL no encoder" >> "$LOG"; exit 1; fi
# belt-and-braces: back the checkpoint up immediately, before any eval can fail
aws s3 sync "$ENC" "s3://climb-s3-bucket/experiments/climb_v2_phase2/$RUN_ID/encoder" --only-show-errors

# --- 2. canonical panels: MoleculeACE, hERG, CBS ---
~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
  --model "$RUN_ID" --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
aws s3 cp --recursive "figure_data/chemeleon_suite/moleculeace/$RUN_ID" \
  "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$RUN_ID" --only-show-errors

~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track polaris --featurizer encoder \
  --model "$RUN_ID" --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
.venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py "figure_data/chemeleon_suite/polaris/$RUN_ID" >> "$LOG" 2>&1
aws s3 cp --recursive "figure_data/chemeleon_suite/polaris/$RUN_ID" \
  "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/$RUN_ID" --only-show-errors

~/venvs/climb/bin/python eval_v2.py --encoder "$ENC" --tokenizer figure_data/_tokenizer \
  --output_dir "figure_data/cbs_benchmark/$RUN_ID/moleculenet_cv" --head mlp --head_seeds 0 1 2 \
  --task_csv data/cbs.csv --task_name cbs --task_type classification \
  --cv_folds 5 --cv_scheme provided >> "$LOG" 2>&1
aws s3 cp --recursive "figure_data/cbs_benchmark/$RUN_ID" \
  "s3://climb-s3-bucket/experiments/cbs_benchmark/$RUN_ID" --only-show-errors

ok=0
[ -f "figure_data/chemeleon_suite/moleculeace/$RUN_ID/verified.json" ] && ok=$((ok+1))
[ -f "figure_data/chemeleon_suite/polaris/$RUN_ID/polaris_scores.csv" ] && ok=$((ok+1))
[ -f "figure_data/cbs_benchmark/$RUN_ID/moleculenet_cv/suite_summary.json" ] && ok=$((ok+1))
echo "[cmtr] panels=$ok/3 $(date -u +%FT%TZ)" >> "$LOG"
if [ "$ok" -eq 3 ]; then
  touch "figure_data/${RUN_ID}_DONE"; echo "[cmtr] done -> shutdown" >> "$LOG"; sudo shutdown -h now
else
  echo "[cmtr] incomplete -> staying UP" >> "$LOG"
fi

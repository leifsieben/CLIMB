#!/usr/bin/env bash
# fig_C2 / fig_D need a FLOOR on every canonical task: "no pretrain, end2end" = a random-init
# encoder fine-tuned on the task. It exists for CBS (e2e_random_00/01/02) and, as of today, for
# MoleculeACE (no_pretrain_e2e_e2e{,_s1,_s2}). It does NOT exist for Ames/Polaris, which caps
# fig_C2 and fig_D at 4 of 6 canonical tasks.
#
# Mirrors the MoleculeACE floor exactly: fine-tune FROM the same saved random_baseline_XX weights
# the frozen control was scored at, so the frozen and e2e bars differ in exactly one thing.
# Polaris withholds test labels, so this is the usual two-step: predict here, score in .venv_polaris.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/ames_e2e_floor.log
say() { echo "[floor] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"

mkdir -p chemeleon_suite/data/polaris
[ -f chemeleon_suite/data/polaris/tdcommons__ames.csv ] || \
  aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

build_polaris_venv() {
  for a in 1 2 3; do
    [ -x .venv_polaris/bin/python ] && .venv_polaris/bin/python -c "import numpy, polaris" 2>/dev/null && return 0
    rm -rf .venv_polaris; python3.12 -m venv .venv_polaris
    .venv_polaris/bin/python -m pip install -q "numpy<2" >> "$LOG" 2>&1
    .venv_polaris/bin/python -m pip install -q "polaris-lib==0.13.0" rdkit scikit-learn >> "$LOG" 2>&1
  done
  .venv_polaris/bin/python -c "import numpy, polaris" 2>/dev/null
}
build_polaris_venv || { say "FATAL polaris venv -> staying UP"; exit 1; }

done_ok() {  # >=20 distinct tasks, never file existence
  local f=figure_data/chemeleon_suite/polaris/$1/polaris_scores.csv
  [ -s "$f" ] && [ "$(tail -n +2 "$f" | cut -d, -f1 | sort -u | wc -l)" -ge 20 ]
}

ok=0; total=3; i=0
for enc_run in random_baseline_00 random_baseline_01 random_baseline_02; do
  case $i in 0) out=no_pretrain_e2e_e2e;; 1) out=no_pretrain_e2e_e2e_s1;; 2) out=no_pretrain_e2e_e2e_s2;; esac
  i=$((i+1))
  if done_ok "$out"; then say "SKIP $out"; ok=$((ok+1)); continue; fi
  ENC=figure_data/_stage_floor/$enc_run/encoder
  if [ ! -f "$ENC/model.safetensors" ]; then
    mkdir -p "$ENC"
    aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$enc_run/encoder" "$ENC" --only-show-errors
  fi
  [ -f "$ENC/model.safetensors" ] || { say "ERROR $out: no encoder $enc_run"; continue; }
  say "polaris e2e floor: $out (from $enc_run)"
  ~/venvs/climb/bin/python scripts/chemeleon_suite_e2e.py --track polaris \
    --model "$out" --suffix "" --encoder "$ENC" --tokenizer figure_data/_tokenizer \
    --seeds 42 117 709 >> "$LOG" 2>&1
  .venv_polaris/bin/python scripts/chemeleon_suite_score_polaris.py \
    "figure_data/chemeleon_suite/polaris/$out" >> "$LOG" 2>&1
  if done_ok "$out"; then
    aws s3 cp --recursive "figure_data/chemeleon_suite/polaris/$out" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/polaris/$out" --only-show-errors
    say "OK $out"; ok=$((ok+1))
  else say "FAIL $out"; fi
  rm -rf "figure_data/_stage_floor/$enc_run"
done
say "DONE $ok/$total"
[ "$ok" -eq "$total" ] && { say "verified -> shutdown"; sudo shutdown -h now; } || say "incomplete -> staying UP"

#!/usr/bin/env bash
# The last replication asymmetry in the suite: MoleculeACE has arms at 1 pretraining seed sitting
# beside arms at 3, so its error bars are not the same quantity across the panel.
#
# Three arms were flagged. Only TWO need compute:
#   s2u_dense       -- NO COMPUTE. All three MoleculeACE dirs already existed (630 rows each); the
#                      resolver missed them because arms.py names the base dir s2u_dense_from8M_s0
#                      while mace_seed_dirs() expands <base>, <base>_s1, <base>_s2 -- so it looked
#                      for ..._s0_s1. Fixed by creating the bare dir, already done.
#   random_encoder  -- frozen probe of random_baseline_01 / _02.
#   e2e_no_pretrain -- e2e fine-tune FROM random_baseline_01 / _02, mirroring run_e2e_random.py and
#                      the original no_pretrain_e2e_e2e (which fine-tuned from random_baseline_00).
#
# Output dirs are named after the ENCODER they came from (random_baseline_01/02), which is truthful
# but does NOT match mace_seed_dirs()'s <base>_s1/_s2 expansion -- arms.py needs an explicit list
# for these arms, exactly as its `mol` field already uses. Flagged rather than papered over by
# giving the dirs misleading names.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/mace_replicates.log
say() { echo "[macerep] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"

[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ "$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)" -ge 30 ] || {
  mkdir -p chemeleon_suite/data/moleculeace
  aws s3 sync s3://climb-s3-bucket/datasets/moleculeace/ chemeleon_suite/data/moleculeace/ --only-show-errors; }

stage() {  # $1 = encoder run
  local enc=figure_data/_stage_macerep/$1/encoder
  [ -f "$enc/model.safetensors" ] || { mkdir -p "$enc"
    aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$1/encoder" "$enc" --only-show-errors; }
  [ -f "$enc/model.safetensors" ] && echo "$enc"
}
done_ok() { [ -s "figure_data/chemeleon_suite/moleculeace/$1/results.csv" ]; }

ok=0; total=4
for r in random_baseline_01 random_baseline_02; do
  if done_ok "$r"; then say "SKIP $r (frozen done)"; ok=$((ok+1)); continue; fi
  ENC=$(stage "$r"); [ -z "$ENC" ] && { say "ERROR $r: no encoder"; continue; }
  say "frozen MoleculeACE: $r"
  ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
    --model "$r" --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp \
    --seeds 42 117 709 >> "$LOG" 2>&1
  if done_ok "$r"; then
    aws s3 cp --recursive "figure_data/chemeleon_suite/moleculeace/$r" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$r" --only-show-errors
    say "OK $r"; ok=$((ok+1))
  else say "FAIL $r"; fi
done

for i in 01 02; do
  out="no_pretrain_e2e_e2e_s${i#0}"
  if done_ok "$out"; then say "SKIP $out (e2e done)"; ok=$((ok+1)); continue; fi
  ENC=$(stage "random_baseline_$i"); [ -z "$ENC" ] && { say "ERROR $out: no encoder"; continue; }
  say "e2e MoleculeACE: $out (fine-tuned from random_baseline_$i)"
  ~/venvs/climb/bin/python scripts/chemeleon_suite_e2e.py --track moleculeace \
    --model "$out" --suffix "" --encoder "$ENC" --tokenizer figure_data/_tokenizer \
    --seeds 42 117 709 >> "$LOG" 2>&1
  if done_ok "$out"; then
    aws s3 cp --recursive "figure_data/chemeleon_suite/moleculeace/$out" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$out" --only-show-errors
    say "OK $out"; ok=$((ok+1))
  else say "FAIL $out"; fi
done

say "DONE $ok/$total"
if [ "$ok" -eq "$total" ]; then say "all verified -> shutdown"; sudo shutdown -h now
else say "incomplete -> staying UP for inspection"; fi

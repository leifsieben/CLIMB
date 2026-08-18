#!/usr/bin/env bash
# SI fig e label-efficiency sweep on the canonical panels. ONE box, sequential, cheapest first.
#
# CBS IS DELIBERATELY NOT HERE. CBS has 43 actives total, so an 80% train split at the sweep's
# fractions leaves ~2 / 3 / 9 / 17 / 34 actives. NEF1@1% computed from 2 actives is noise, not a
# low-label data point, and the crossover it would show would be an artefact of losing the positive
# class rather than of label budget. This was already reasoned through when cbs_e2e.py was written
# (see its docstring: "subsampling train strips the positive signal ... the low-fraction points are
# noise"). Running it would put a panel on the figure whose x-axis means something different from
# the others' -- exactly what we were asked to prevent. Raised rather than silently run.
#
# So this covers Ames (cheaper) then MoleculeACE (~60% of the cost, last so a lost box costs the
# expensive panel and the figure still gains a panel).
#
# CONSISTENCY: same three arms as the existing panels, with the two pretrained arms as FROZEN
# PROBES ON THE SAME ENCODERS fig_A plots (unsup_8M, skip_dense_8M) -- not fresh pretraining; same
# 5 fractions applied PER TASK; each benchmark's NATIVE split (Ames = Polaris split, MoleculeACE =
# its 30 targets' own train/test). Test sets are never subsampled.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/si_fige.log
say() { echo "[fige-le] $(date -u +%FT%TZ) $*" >> "$LOG"; }
FRACTIONS="0.05 0.10 0.25 0.50 1.00"
say "start; fractions=$FRACTIONS; panels=Ames,MoleculeACE (CBS excluded: only 43 actives)"

[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ "$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)" -ge 30 ] || {
  mkdir -p chemeleon_suite/data/moleculeace
  aws s3 sync s3://climb-s3-bucket/datasets/moleculeace/ chemeleon_suite/data/moleculeace/ --only-show-errors; }
[ -f chemeleon_suite/data/polaris/tdcommons__ames.csv ] || {
  mkdir -p chemeleon_suite/data/polaris
  aws s3 sync s3://climb-s3-bucket/datasets/polaris/ chemeleon_suite/data/polaris/ --only-show-errors; }

stage() {
  local enc=figure_data/_stage_fige_le/$1/encoder
  [ -f "$enc/model.safetensors" ] || { mkdir -p "$enc"
    aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$1/encoder" "$enc" --only-show-errors; }
  [ -f "$enc/model.safetensors" ] && echo "$enc"
}
UNSUP=$(stage unsup_8M); SUPD=$(stage skip_dense_8M); RAND=$(stage random_baseline_00)
for v in "$UNSUP" "$SUPD" "$RAND"; do [ -z "$v" ] && { say "FATAL missing encoder -> staying UP"; exit 1; }; done
TOK=figure_data/_tokenizer

sweep() {   # $1 = track (polaris|moleculeace), $2 = output prefix
  local track=$1 pfx=$2 f arm ENC m
  local subdir; [ "$track" = polaris ] && subdir=polaris || subdir=moleculeace
  local probe;  [ "$track" = polaris ] && probe=test_predictions.csv || probe=results.csv
  for f in $FRACTIONS; do
    for arm in unsup sup_dense e2e; do
      case $arm in unsup) ENC=$UNSUP;; sup_dense) ENC=$SUPD;; e2e) ENC=$RAND;; esac
      m="${pfx}_${arm}_f${f}"
      if [ -s "figure_data/chemeleon_suite/$subdir/$m/$probe" ]; then say "SKIP $m"; continue; fi
      say "$track $arm frac=$f -> $m"
      if [ "$arm" = e2e ]; then
        ~/venvs/climb/bin/python scripts/chemeleon_suite_e2e.py --track "$track" --model "$m" \
          --suffix "" --encoder "$ENC" --tokenizer "$TOK" --seeds 42 117 709 \
          --train_fraction "$f" --subsample_seed 0 >> "$LOG" 2>&1 || say "  FAILED $m"
      else
        ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track "$track" \
          --featurizer encoder --model "$m" --encoder "$ENC" --tokenizer "$TOK" --head mlp \
          --seeds 42 117 709 --train_fraction "$f" --subsample_seed 0 >> "$LOG" 2>&1 || say "  FAILED $m"
      fi
    done
    aws s3 cp --recursive "figure_data/chemeleon_suite/$subdir" \
      "s3://climb-s3-bucket/experiments/chemeleon_suite/$subdir" --exclude "*" --include "${pfx}_*" --only-show-errors
  done
}

sweep polaris     le_ames    ; say "Ames panel done"
sweep moleculeace le_mace    ; say "MoleculeACE panel done"

na=$(ls -d figure_data/chemeleon_suite/polaris/le_ames_* 2>/dev/null | wc -l)
nm=$(ls -d figure_data/chemeleon_suite/moleculeace/le_mace_* 2>/dev/null | wc -l)
say "DONE ames=$na/15 mace=$nm/15"
if [ "$na" -ge 15 ] && [ "$nm" -ge 15 ]; then say "both panels complete -> shutdown"; sudo shutdown -h now
else say "PARTIAL (ames=$na mace=$nm) -> staying UP"; fi

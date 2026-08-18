#!/usr/bin/env bash
# Is chemeleon_frozen's QM7 = 268.8 real, or a bad draw?
#
# WHAT THE EXISTING RUN ALREADY ANSWERS (no compute needed): it is NOT head-init noise. That run
# already carries 3 head seeds x 5 folds, and fold2 blows up on ALL THREE -- 369.99 / 529.39 /
# 404.25 against ~210-250 on folds 0/1/4 (fold3 is elevated too: 262 / 348 / 265). So more head
# seeds would only re-confirm what is already there. CheMeleon's embedding is deterministic and
# external, so there is no pretraining seed to vary either.
#
# THE INFORMATIVE VARIATION IS THE FOLD PARTITION. --subsample_seed also seeds
# _scaffold_kfold_indices (eval_v2.py:479), so re-partitioning asks the question that matters:
# does the blow-up follow the MOLECULES (a scaffold group CheMeleon embeds badly -> real,
# reportable failure on atomization energy) or the fold INDEX (an artifact of one partition)?
#
# Runs both, cheaply:
#   A. head-seed replicates 3-5 and 6-8, giving the arm the 3 seed dirs the convention expects
#   B. fold-partition seeds 1 and 2 -- the actual diagnostic
# QM7 only. Non-destructive: every output goes to its own dir, never over chemeleon_frozen/.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/chemeleon_frozen_seeds.log
say() { echo "[cfz] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"
# The frozen CheMeleon probe needs chemprop's CheMeleonFingerprint, which is NOT in the climb
# venv. Build the same venv the other CheMeleon jobs use, and VERIFY the import before running --
# otherwise the run fails 30 minutes in on a ModuleNotFoundError.
PY=~/venvs/chemeleon/bin/python
if ! $PY -c "from chemeleon_fingerprint import CheMeleonFingerprint" 2>/dev/null; then
  say "building chemeleon venv"
  python3.12 -m venv ~/venvs/chemeleon
  $PY -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  $PY -m pip install -q "chemprop==2.3.1" rdkit deepchem==2.5.0 xgboost >> "$LOG" 2>&1
  [ -f scripts/molnet_box_bootstrap.sh ] && bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi
if ! $PY -c "from chemeleon_fingerprint import CheMeleonFingerprint" 2>/dev/null; then
  say "FATAL chemeleon venv unusable -> staying UP"; exit 1
fi
say "python=$PY (CheMeleonFingerprint import verified)"

run_one() {  # $1 = out dir suffix, $2..$4 = head seeds, $5 = subsample/partition seed
  local name=$1 s1=$2 s2=$3 s4=$4 psd=$5
  local out="figure_data/climb_v2_phase2/${name}/moleculenet_cv"
  if [ -s "$out/moleculenet_summary.csv" ]; then say "SKIP $name"; return 0; fi
  say "run $name (head seeds $s1 $s2 $s4, partition seed $psd)"
  $PY eval_v2.py --output_dir "$out" --datasets QM7 --featurizer chemeleon --head mlp \
    --head_seeds "$s1" "$s2" "$s4" --cv_folds 5 --cv_scheme scaffold \
    --subsample_seed "$psd" >> "$LOG" 2>&1
  if [ -s "$out/moleculenet_summary.csv" ]; then
    aws s3 cp --recursive "figure_data/climb_v2_phase2/${name}" \
      "s3://climb-s3-bucket/experiments/climb_v2_phase2/${name}" --only-show-errors
    say "OK $name  QM7 mean=$(awk -F, '$1=="QM7" && $7=="rmse" && $8=="MEAN"{print $10}' "$out/moleculenet_summary.csv")"
    return 0
  fi
  say "FAIL $name"; return 1
}

ok=0
run_one chemeleon_frozen_s1        3 4 5 0 && ok=$((ok+1))   # A: head-seed replicate 2
run_one chemeleon_frozen_s2        6 7 8 0 && ok=$((ok+1))   # A: head-seed replicate 3
run_one chemeleon_frozen_part1     0 1 2 1 && ok=$((ok+1))   # B: re-partitioned CV
run_one chemeleon_frozen_part2     0 1 2 2 && ok=$((ok+1))   # B: re-partitioned CV

say "DONE $ok/4"
if [ "$ok" -eq 4 ]; then say "all verified -> shutdown"; sudo shutdown -h now
else say "incomplete -> staying UP for inspection"; fi

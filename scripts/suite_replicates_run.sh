#!/usr/bin/env bash
# Three model-seed replicates on MoleculeACE and Polaris for the four arms that still have one.
#
# WHY. Every CLIMB arm resolves to THREE dirs on both suite tracks (skip_dense_8M/_s1/_s2, ...),
# each itself a 3-seed fit, so a CLIMB bar on MoleculeACE or Ames is 9 fits. Four arms resolve to
# ONE dir and so to 3 fits:
#
#     ecfp4, fp_desc          deterministic featurizers -- no pretraining stage to reseed
#     chemeleon_frozen        a fixed external checkpoint -- likewise
#     chemeleon_e2e           one fine-tuning run
#
# The peer session already closed the same gap on MoleculeNet and CBS (commit 3c52686) by giving
# each of those arms two more replicates on DISJOINT head-seed triples. That is the apples-to-apples
# move available here: an ECFP fingerprint has no pretraining seed and never will, so the axis that
# CAN be equalised is the number of independent FITS behind the bar. This script does the same for
# the two suite tracks, which were left out.
#
# Seed triples are disjoint from the mainline's {42,117,709} and from each other, so the three dirs
# are independent draws rather than three views of one fit.
#
# NOTE ON COST: ecfp4/fp_desc/chemeleon_frozen are frozen-probe fits (featurize once, fit a head)
# and are cheap. chemeleon_e2e RE-FINE-TUNES a D-MPNN on 30 MoleculeACE targets and 28 Polaris
# tasks per replicate and is the expensive one -- it is last so the cheap three land first if the
# box is interrupted.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/suite_replicates.log
CLIMB_PY=~/venvs/climb/bin/python
CHEM_PY=~/venvs/chemeleon/bin/python
S3=s3://climb-s3-bucket/experiments
say () { echo "[reps] $* $(date -u +%FT%TZ)" >> "$LOG"; }
say "start"

if [ ! -x "$CHEM_PY" ]; then
  say "building chemeleon venv"
  python3.12 -m venv ~/venvs/chemeleon
  $CHEM_PY -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  $CHEM_PY -m pip install -q "chemprop==2.3.1" xgboost rdkit deepchem==2.5.0 >> "$LOG" 2>&1
  bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi

rep () {   # $1 py  $2 base model  $3 suffix  $4 seed triple  $5.. featurizer args
  local PY=$1 base=$2 sfx=$3 seeds=$4; shift 4
  for track in moleculeace polaris; do
    local out="figure_data/chemeleon_suite/$track/${base}${sfx}"
    local marker="results.csv"; [ "$track" = polaris ] && marker="polaris_scores.csv"
    if [ -s "$out/$marker" ]; then say "SKIP $track ${base}${sfx}"; continue; fi
    $PY scripts/chemeleon_suite_run.py --track $track --model "${base}${sfx}" \
        --seeds $seeds --head mlp "$@" >> "$LOG" 2>&1
    say "$track ${base}${sfx} rc=$?"
    [ -s "$out/$marker" ] && aws s3 cp --recursive "$out" "$S3/chemeleon_suite/$track/${base}${sfx}" --only-show-errors
  done
}

# --- cheap: frozen probes over a deterministic or fixed featurizer --------------------------------
rep "$CLIMB_PY" ecfp4            _s1 "43 118 710"  --featurizer ecfp4   --head xgb
rep "$CLIMB_PY" ecfp4            _s2 "44 119 711"  --featurizer ecfp4   --head xgb
rep "$CLIMB_PY" fp_desc          _s1 "43 118 710"  --featurizer fp_desc --head xgb
rep "$CLIMB_PY" fp_desc          _s2 "44 119 711"  --featurizer fp_desc --head xgb
rep "$CHEM_PY"  chemeleon_frozen _s1 "43 118 710"  --featurizer chemeleon
rep "$CHEM_PY"  chemeleon_frozen _s2 "44 119 711"  --featurizer chemeleon

# --- expensive: re-fine-tunes the whole D-MPNN, 58 tasks per replicate ----------------------------
# chemeleon_e2e goes through chemeleon_e2e_gaps.py (chemprop CLI), not chemeleon_suite_run.py.
# It is env-driven: CHEM_SEEDS picks the triple and CHEM_RUN the output dir -- the latter added
# 2026-08-19, because the run name was hardcoded and a second invocation would have overwritten
# the published dir instead of sitting beside it.
if [ -x "$CHEM_PY" ]; then
  for spec in "_s1 43 118 710" "_s2 44 119 711"; do
    set -- $spec
    sfx=$1; triple="$2 $3 $4"
    if [ -s "figure_data/chemeleon_suite/moleculeace/chemeleon_e2e${sfx}/results.csv" ] \
       && [ -s "figure_data/chemeleon_suite/polaris/chemeleon_e2e${sfx}/polaris_scores.csv" ]; then
      say "SKIP chemeleon_e2e${sfx}"; continue
    fi
    CHEM_RUN="chemeleon_e2e${sfx}" CHEM_SEEDS="$triple" CHEM_ONLY=mace \
      $CHEM_PY scripts/chemeleon_e2e_gaps.py >> "$LOG" 2>&1
    say "moleculeace chemeleon_e2e${sfx} rc=$?"
    CHEM_RUN="chemeleon_e2e${sfx}" CHEM_SEEDS="$triple" CHEM_ONLY=polaris_all \
      $CHEM_PY scripts/chemeleon_e2e_gaps.py >> "$LOG" 2>&1
    say "polaris chemeleon_e2e${sfx} rc=$?"
    # Polaris withholds test labels, so its scoring runs in the polaris-lib venv, as in
    # scripts/chemeleon_e2e_polaris_run.sh
    D="figure_data/chemeleon_suite/polaris/chemeleon_e2e${sfx}"
    [ -x .venv_polaris/bin/python ] && .venv_polaris/bin/python \
        scripts/chemeleon_suite_score_polaris.py "$D" >> "$LOG" 2>&1
    for t in moleculeace polaris; do
      d="figure_data/chemeleon_suite/$t/chemeleon_e2e${sfx}"
      [ -d "$d" ] && aws s3 cp --recursive "$d" "$S3/chemeleon_suite/$t/chemeleon_e2e${sfx}" --only-show-errors
    done
  done
fi

aws s3 cp "$LOG" $S3/_logs/suite_replicates.log --only-show-errors
n=$(ls -d figure_data/chemeleon_suite/moleculeace/{ecfp4,fp_desc,chemeleon_frozen,chemeleon_e2e}_s[12] 2>/dev/null | wc -l)
say "produced $n of 8 moleculeace replicate dirs"
if [ "$n" -ge 8 ]; then if [ -n "${CHAIN_NEXT:-}" ]; then
    say "complete -> chaining to $CHAIN_NEXT"; exec bash "$CHAIN_NEXT"
  else say "complete -> shutdown"; sudo shutdown -h now; fi
else say "INCOMPLETE -> staying UP for inspection"; fi

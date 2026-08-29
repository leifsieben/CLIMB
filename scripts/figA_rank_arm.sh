#!/usr/bin/env bash
# Produce the RANKING (fig_A) artifacts for one CLIMB arm, beyond what the fig_B battery makes.
#
# figB_eval_run.sh covers two of fig_A's six suites -- MoleculeNet (moleculenet_cv/) and
# MoleculeACE -- and deliberately trims Polaris to tdcommons/ames, because fig_B's Ames panel is
# that one task. fig_A is NOT fig_B: it ranks over MoleculeNet, MoleculeACE, Polaris (all 28
# tasks), CBS, Wong and FartDB, and an arm short of a suite is rescaled rather than dropped, so a
# missing suite silently changes that arm's headline rank instead of failing.
#
# This script fills the gap: full Polaris, CBS, Wong, FartDB.
#
# Usage: figA_rank_arm.sh <run_id>
set -u
set -o pipefail
ARM=$1
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket
LOG=analysis/figA_rank_${ARM}.log
mkdir -p analysis figure_data
say () { echo "[figA:$ARM] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/experiments/figA_clms/logs/figA_rank_${ARM}.log" --only-show-errors; exit 1; }

# ---- the evaluation environment must be the ladder's --------------------------------------------
# Same defect as the fig_B battery: this AMI ships rdkit-pypi 2022.9.5 alongside the pinned rdkit
# 2025.9.2 and `import rdkit` resolves to the shadowing one. Purge, reinstall, then test the
# CANONICALIZATION rather than the version string -- a version string can agree while the import
# resolves elsewhere. The two dists share the same rdkit/ directory, so uninstalling alone leaves a
# half-populated package whose Chem has no MolToSmiles.
if $PY -m pip show rdkit-pypi >/dev/null 2>&1; then
  say "rdkit-pypi is shadowing rdkit -- purging and reinstalling the pinned rdkit"
  $PY -m pip uninstall -y rdkit-pypi 2>&1 | tail -2 | tee -a "$LOG"
  $PY -m pip install -q --force-reinstall --no-deps "rdkit==2025.9.2" 2>&1 | tail -2 | tee -a "$LOG"
fi
$PY - <<'PYCHK' 2>&1 | tee -a "$LOG"
import sys
import rdkit
from rdkit import Chem
canon = Chem.MolToSmiles(Chem.MolFromSmiles("C[Hg]Cl"))
print(f"[figA] rdkit {rdkit.__version__} canonicalizes C[Hg]Cl as {canon}")
if canon != "[CH3][Hg][Cl]":
    sys.exit("rdkit canonicalization is not the reference environment's")
PYCHK
[ "${PIPESTATUS[0]}" = "0" ] || abort "rdkit is not the reference environment"

# ---- staging -------------------------------------------------------------------------------------
ENC=figure_data/climb_v2_phase2/$ARM/encoder
mkdir -p "$ENC"
[ -s "$ENC/model.safetensors" ] || aws s3 cp "$S3/experiments/climb_v2_phase2/$ARM/encoder" "$ENC" --recursive --only-show-errors
[ -s "$ENC/model.safetensors" ] || abort "no encoder weights for $ARM"
[ -f figure_data/_tokenizer/tokenizer.json ] || aws s3 sync $S3/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || abort "tokenizer missing after staging"
say "encoder and tokenizer staged"

# ---- Polaris: ALL 28 TASKS ------------------------------------------------------------------------
# figB_eval_run.sh OVERWRITES this file with the single ames line, so on any box that has run the
# fig_B battery the on-disk list is the trimmed one. Restore it from git and assert the count --
# running the trimmed list here would produce a 1-task Polaris cell that reads as a complete suite.
git checkout -- chemeleon_suite/tasks/polaris_tasks.txt 2>/dev/null || true
NP=$(grep -c . chemeleon_suite/tasks/polaris_tasks.txt)
[ "$NP" -ge 28 ] || abort "polaris task list has $NP tasks, expected 28 -- refusing to write a partial Polaris cell"
say "polaris task list restored: $NP tasks"

pol=figure_data/chemeleon_suite/polaris/$ARM
if [ ! -s "$pol/test_predictions.csv" ] || [ "$(awk -F, 'NR>1{print $1}' "$pol/test_predictions.csv" | sort -u | wc -l | tr -d ' ')" != "$NP" ]; then
  say "polaris: $NP tasks x 3 seeds"
  $PY scripts/chemeleon_suite_run.py --track polaris --featurizer encoder --model "$ARM" \
      --encoder "$ENC" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 \
      2>&1 | tail -15 | tee -a "$LOG"
  got=$(awk -F, 'NR>1{print $1}' "$pol/test_predictions.csv" 2>/dev/null | sort -u | wc -l | tr -d ' ')
  [ "$got" = "$NP" ] || abort "polaris produced $got of $NP tasks"
  aws s3 cp --recursive "$pol" "$S3/experiments/chemeleon_suite/polaris/$ARM" --only-show-errors || abort "polaris upload failed"
  say "polaris done and uploaded ($got tasks) -- still needs the OFF-BOX scoring pass for polaris_scores.csv"
else
  say "polaris already complete -- skipping"
fi

# ---- CBS ------------------------------------------------------------------------------------------
cbs=figure_data/cbs_benchmark/$ARM/moleculenet_cv
if [ ! -s "$cbs/moleculenet_summary.csv" ]; then
  say "cbs"
  $PY scripts/cbs_run.py --model "$ARM" --featurizer encoder --encoder "$ENC" \
      --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 2>&1 | tail -15 | tee -a "$LOG"
  [ -s "$cbs/moleculenet_summary.csv" ] || abort "cbs produced no moleculenet_summary.csv"
  grep -qi 'nef1' "$cbs/moleculenet_summary.csv" || abort "cbs summary carries no nef1 rows -- the panel value cannot be read from it"
  aws s3 cp --recursive "figure_data/cbs_benchmark/$ARM" "$S3/experiments/cbs_benchmark/$ARM" --only-show-errors || abort "cbs upload failed"
  say "cbs done and uploaded"
else
  say "cbs already present -- skipping"
fi

# ---- Wong + FartDB --------------------------------------------------------------------------------
# figA_one_arm.sh already stages the Wong CSV, skips cells that exist on S3, and uploads its own.
say "wong + fartdb via figA_one_arm.sh"
bash scripts/figA_one_arm.sh "$ARM" 2>&1 | tail -20 | tee -a "$LOG" || abort "wong/fartdb battery failed"

# ---- completion is per-artifact -------------------------------------------------------------------
missing=0
check () { [ -s "$1" ] || { say "MISSING $1"; missing=$((missing+1)); }; }
check "figure_data/chemeleon_suite/polaris/$ARM/test_predictions.csv"
check "figure_data/cbs_benchmark/$ARM/moleculenet_cv/moleculenet_summary.csv"
check "figure_data/wong_saureus/$ARM/verified.json"
check "figure_data/fartdb/$ARM/verified.json"
aws s3 cp "$LOG" "$S3/experiments/figA_clms/logs/figA_rank_${ARM}.log" --only-show-errors
if [ "$missing" -eq 0 ]; then
  say "ALL RANKING ARTIFACTS PRESENT for $ARM"
  say "NOTE: Polaris still needs scripts/chemeleon_suite_score_polaris.py OFF-BOX to write polaris_scores.csv"
  [ "${EVAL_SHUTDOWN:-0}" = "1" ] && { say "EVAL_SHUTDOWN=1 -- shutting down"; sudo shutdown -h now; }
  say "EVAL_SHUTDOWN unset -- staying up"
else
  abort "$missing artifact(s) missing"
fi

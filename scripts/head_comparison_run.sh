#!/usr/bin/env bash
# SI head comparison -- MLP probe vs XGBoost probe on the SAME frozen features.
#
# WHY. Every arm in the paper carries exactly ONE head: the classical anchors (ECFP, ECFP+desc)
# were run with XGBoost and every learned encoder with the 2-layer MLP. So "CLIMB loses to
# ECFP+desc" and "CheMeleon frozen loses to ECFP+desc" are each confounded with the head, and
# nothing in the existing data separates representation from probe. This job fills the missing
# half of the 4x2 grid so the confound is measured instead of assumed:
#
#     model              already has   this job adds
#     ECFP+desc          xgb           mlp
#     CLIMB sup, desc    mlp           xgb
#     CLIMB unsup        mlp           xgb
#     CheMeleon frozen   mlp           xgb
#
# across all four suites: MoleculeNet (6 CV tasks), CBS (1), MoleculeACE (30), Polaris (28).
#
# EVERY OTHER KNOB IS COPIED FROM THE MAINLINE RUN that produced the arm's existing half --
# featurizer, pool, standardize, head seeds, dataset list, CV scheme, partition seed -- so the
# paired MLP-vs-XGB difference is attributable to the head and to nothing else. The invocations
# below are lifted from scripts/cv_seed_replicates.sh (MolNet), scripts/cbs_seeds_run.sh and
# scripts/cbs_chemprop_chain.sh (CBS) and scripts/concat_chemeleon_run.sh (the suite tracks).
# TASKS excludes Lipophilicity for the same reason cv_seed_replicates.sh does.
#
# Outputs land in <run>__<head>/ so a mainline directory can never be overwritten; the figure
# reader pairs "<run>" with "<run>__<head>".
#
# Runs on climb-a2-bootstrap, c5.4xlarge (user decision 2026-08-19). Self-stops only when the full
# grid is on disk; stays up otherwise so the log is inspectable.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis figure_data
LOG=analysis/head_comparison.log
CLIMB_PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments
TASKS="ESOL QM7 BBBP BACE Tox21 HIV"
say () { echo "[head] $* $(date -u +%FT%TZ)" >> "$LOG"; }
say "start"

[ -f figure_data/_tokenizer/tokenizer.json ] || {
  mkdir -p figure_data/_tokenizer
  aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }

# chemprop venv supplies the CheMeleon fingerprint featurizer (recipe from concat_chemeleon_run.sh)
if [ ! -x ~/venvs/chemeleon/bin/python ]; then
  say "building chemeleon venv"
  python3.12 -m venv ~/venvs/chemeleon
  ~/venvs/chemeleon/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
  ~/venvs/chemeleon/bin/python -m pip install -q "chemprop==2.3.1" xgboost rdkit deepchem==2.5.0 >> "$LOG" 2>&1
  bash scripts/molnet_box_bootstrap.sh ~/venvs/chemeleon >> "$LOG" 2>&1
fi

# The suite task CSVs are DATA, and scripts/deploy_to_ec2.sh ships code only -- so on a fresh box
# chemeleon_suite_run.py dies with FileNotFoundError on the first MoleculeACE task. Staged here
# rather than assumed present, and asserted, because a missing dataset should stop the job at the
# top instead of failing 30 tasks in a row further down.
aws s3 sync s3://climb-s3-bucket/datasets/moleculeace/ chemeleon_suite/data/moleculeace/ --only-show-errors
aws s3 sync s3://climb-s3-bucket/datasets/polaris/    chemeleon_suite/data/polaris/    --only-show-errors
nmace=$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)
npol=$(ls chemeleon_suite/data/polaris/*.csv 2>/dev/null | wc -l)
say "staged suite data: $nmace moleculeace, $npol polaris"
if [ "$nmace" -lt 30 ] || [ "$npol" -lt 28 ]; then
  say "FATAL suite data incomplete (want >=30 / >=28) -- staying UP"; exit 1
fi

stage_encoder () {   # $1 = run name under climb_v2_phase2 -> echoes the local encoder path
  local d=figure_data/_stage_head/$1/encoder
  [ -f $d/model.safetensors ] || { mkdir -p $d
    aws s3 sync $S3/climb_v2_phase2/$1/encoder $d --only-show-errors >> "$LOG" 2>&1; }
  echo $d
}

# ---- one (model, head) cell across all four suites ---------------------------------------------
# $1 py  $2 out-tag  $3 head  $4 mace/polaris model name  $5.. featurizer + encoder args
# EVAL_ONLY carries flags eval_v2 accepts and chemeleon_suite_run.py does not (--features_npz);
# passing them to both would abort the suite tracks with "unrecognized arguments". Reset per call
# by the caller.
EVAL_ONLY=""
cell () {
  local PY=$1 tag=$2 head=$3 suitemodel=$4; shift 4
  local out="figure_data/climb_v2_phase2/${tag}__${head}/moleculenet_cv"
  if [ -s "$out/moleculenet_summary.csv" ]; then say "SKIP molnet ${tag}__${head}"; else
    $PY eval_v2.py --output_dir "$out" --datasets $TASKS --head $head --head_seeds 0 1 2 \
        --pool mean --standardize zscore --cv_folds 5 --cv_scheme scaffold \
        $EVAL_ONLY "$@" >> "$LOG" 2>&1
    say "molnet ${tag}__${head} rc=$?"
  fi

  local cout="figure_data/cbs_benchmark/${tag}__${head}/moleculenet_cv"
  if [ -s "$cout/moleculenet_summary.csv" ]; then say "SKIP cbs ${tag}__${head}"; else
    $PY eval_v2.py --output_dir "$cout" --head $head --head_seeds 0 1 2 \
        --pool mean --standardize zscore --cv_folds 5 --cv_scheme provided \
        --task_csv data/cbs.csv --task_name cbs --task_type classification \
        $EVAL_ONLY "$@" >> "$LOG" 2>&1
    say "cbs ${tag}__${head} rc=$?"
  fi

  for track in moleculeace polaris; do
    local sout="figure_data/chemeleon_suite/$track/${suitemodel}__${head}"
    if [ -d "$sout" ] && [ -n "$(ls -A $sout 2>/dev/null)" ]; then say "SKIP $track ${suitemodel}__${head}"; continue; fi
    $PY scripts/chemeleon_suite_run.py --track $track --model "${suitemodel}__${head}" \
        --head $head --seeds 42 117 709 "$@" >> "$LOG" 2>&1
    say "$track ${suitemodel}__${head} rc=$?"
  done
}

# 1. the classical anchor gets the MLP it never had
cell "$CLIMB_PY" fp_desc_anchor mlp fp_desc --featurizer fp_desc

# 2. the two CLIMB encoders get XGBoost
for m in skip_dense_8M unsup_8M; do
  E=$(stage_encoder $m)
  [ -f "$E/model.safetensors" ] || { say "MISSING encoder $m -- skipped"; continue; }
  cell "$CLIMB_PY" "$m" xgb "$m" --featurizer encoder --encoder "$E" --tokenizer figure_data/_tokenizer
done

# 3. CheMeleon frozen gets XGBoost.
#
# SPLIT FEATURIZATION, not --featurizer chemeleon in the 3.12 venv. CheMeleon needs
# chemprop>=2.2 -> Python>=3.11, and deepchem 2.8.0 -- the version that DEFINES our Tox21 parse --
# has no 3.12 wheel, so the two cannot share an interpreter. A 3.12 box parses 7,831 Tox21
# molecules where the reference environment parses 7,823, and the resulting ~0.008 ROC-AUC drift
# would land on the XGB half of ONE arm while its MLP half (already on disk, 77,864 reference
# rows) stayed clean -- i.e. it would appear as a head effect. So the box only turns strings into
# vectors and every parsing, fold and scoring decision stays in the reference venv.
# figure_data/_chemeleon_features.npz already covers BACE/Tox21/HIV/cbs (peer session); this
# extends it to the full MolNet CV list so one table serves every eval_v2 call here.
NPZ=figure_data/_chemeleon_head.npz
if [ ! -f "$NPZ" ]; then
  $CLIMB_PY scripts/export_task_smiles.py $TASKS --out figure_data/_task_smiles_head.json >> "$LOG" 2>&1
  say "exported SMILES rc=$?"
  ~/venvs/chemeleon/bin/python scripts/embed_chemeleon_box.py       figure_data/_task_smiles_head.json "$NPZ" >> "$LOG" 2>&1
  say "embedded CheMeleon vectors rc=$?"
fi
# the suite tracks keep --featurizer chemeleon: MoleculeACE and Polaris use their own loaders,
# not deepchem's, so the Tox21 parse difference does not reach them, and their MLP half was
# produced the same way.
EVAL_ONLY="--features_npz $NPZ"
cell "$CLIMB_PY" chemeleon_frozen xgb chemeleon_frozen --featurizer chemeleon
EVAL_ONLY=""

# ---- ship everything back ----------------------------------------------------------------------
for base in climb_v2_phase2 cbs_benchmark; do
  for d in figure_data/$base/*__*; do
    [ -d "$d" ] && aws s3 cp --recursive "$d" $S3/$base/$(basename "$d") --only-show-errors
  done
done
for t in moleculeace polaris; do
  for d in figure_data/chemeleon_suite/$t/*__*; do
    [ -d "$d" ] && aws s3 cp --recursive "$d" $S3/chemeleon_suite/$t/$(basename "$d") --only-show-errors
  done
done
aws s3 cp "$LOG" $S3/_logs/head_comparison.log --only-show-errors

# 4 models x (MolNet + CBS) = 8 eval_v2 output dirs when the grid is complete
n=$(ls -d figure_data/climb_v2_phase2/*__* figure_data/cbs_benchmark/*__* 2>/dev/null | wc -l)
say "produced $n eval_v2 output dir(s) of 8"
if [ "$n" -ge 8 ]; then if [ -n "${CHAIN_NEXT:-}" ]; then
    say "complete -> chaining to $CHAIN_NEXT"; exec bash "$CHAIN_NEXT"
  else say "complete -> shutdown"; sudo shutdown -h now; fi
else say "INCOMPLETE -> staying UP for inspection"; fi

#!/usr/bin/env bash
# 5-fold CV for the pretraining-seed REPLICATES, so every matched-budget arm has the same shape.
#
# The problem this closes. The 8M matched budget has three pretraining seeds per arm, but only
# seed 0 was ever evaluated under 5-fold CV; seeds 1 and 2 have the single scaffold hold-out only.
# So a "3 pretraining seeds x 5 folds = 15 points" comparison was not computable for ANY arm, and
# any panel mixing a 3-seed arm with a 1-seed arm was comparing different quantities.
#
# 22 runs: unsup_only s1/s2, sup_only x5 recipes x s1/s2, unsup->sup x5 recipes x s1/s2.
# The anchors (ecfp4, fp_desc) already have CV; they gain their per-(seed,fold) cells for free from
# the eval_v2 change, since re-running is not needed to decompose an existing average -- they are
# re-run here anyway so every model in the comparison comes from one code version.
#
# Idempotent: a run whose CV summary already carries the per-cell rows is skipped, so this resumes.
set -uo pipefail
cd /home/ec2-user/CLIMB 2>/dev/null || cd /Users/lsieben/VSCode/CLIMB
PY=${PY:-/home/ec2-user/venvs/climb/bin/python}
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2
ROOT=${ROOT:-figure_data/climb_v2_phase2}
TOK=${TOK:-experiments/_tok_cvrep}
TASKS=${TASKS:-"ESOL QM7 BBBP BACE Tox21 HIV"}     # Lipophilicity deliberately excluded: the eval
                                                   # blocklist predates it, so it is not deduped
RECIPES="dense sparse_all dense_plus_sparse minimol_full mixed"
# Seed 0 is included deliberately. Its CV already exists and its fold values are comparable (the
# only eval_v2 changes since are additive), but it was scored before the per-(seed,fold) rows
# existed -- so within one arm seed 0 would carry a different SCHEMA from seeds 1 and 2. Re-running
# it costs one extra pass and makes every run behind an A1 bar come from one code version.
RUNS="unsup_8M unsup_8M_s1 unsup_8M_s2"
for r in $RECIPES; do
  RUNS="$RUNS skip_${r}_8M skip_${r}_8M_s1 skip_${r}_8M_s2"
  RUNS="$RUNS u2s_${r}_from8M u2s_${r}_from8M_s1 u2s_${r}_from8M_s2"
done
RUNS="$RUNS ecfp4_anchor fp_desc_anchor"

aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors
say(){ echo "[cvrep $(date -u +%H:%M:%S)] $*"; }
say "$(echo $RUNS | wc -w) runs queued"

for name in $RUNS; do
    d="$ROOT/$name"
    sum="$d/moleculenet_cv/moleculenet_summary.csv"
    if [ -f "$sum" ] && grep -q "_cell" "$sum" 2>/dev/null; then
        say "$name: per-(seed,fold) cells already present - skipping"; continue
    fi
    mkdir -p "$d"
    is_anchor=0; case "$name" in *anchor) is_anchor=1;; esac
    if [ "$is_anchor" -eq 0 ]; then
        aws s3 sync "$S3/$name/encoder" "$d/encoder" --only-show-errors
        if [ ! -f "$d/encoder/model.safetensors" ]; then
            say "$name: NO ENCODER in S3 - skipping (cannot evaluate)"; continue
        fi
        say "$name: 5-fold CV"
        $PY eval_v2.py --output_dir "$d/moleculenet_cv" --encoder "$d/encoder" --tokenizer "$TOK" \
            --pool mean --standardize zscore --head mlp --max_length 256 \
            --head_seeds 0 1 2 --cv_folds 5 --subsample_seed 0 --datasets $TASKS
    else
        feat=ecfp4; case "$name" in fp_desc*) feat=fp_desc;; esac
        say "$name: 5-fold CV (classical anchor, --featurizer $feat)"
        $PY eval_v2.py --output_dir "$d/moleculenet_cv" --featurizer "$feat" --head xgb \
            --head_seeds 0 1 2 --cv_folds 5 --subsample_seed 0 --datasets $TASKS
    fi
    rc=$?
    # Upload immediately: a box that dies mid-sweep must not take finished evaluations with it.
    aws s3 sync "$d/moleculenet_cv" "$S3/$name/moleculenet_cv" --only-show-errors
    say "$name: rc=$rc"
done
say "CVREP_DONE"

#!/usr/bin/env bash
# 5-fold scaffold CV evaluation for every completed run in a wave, on the box that trained it.
#
# The wave worker only produces the single DeepChem scaffold hold-out eval. For H1 that is the
# wrong headline: the sweep varies the unique-molecule fraction, and the hold-out puts the rarest
# scaffolds in a small test set, so the between-arm differences it reports are smaller than the
# split noise. 5-fold CV uses every molecule as test exactly once and gives a real split-variance
# error bar, which is what a canonical-vs-enumerated comparison needs to say anything at all.
#
# Runs as the guard's POST_HOOK, i.e. after training + the single-split eval and BEFORE the guard
# uploads and stops the box. Idempotent: a run whose CV summary already names HIV is skipped, so
# an interrupted pass resumes.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
MANIFEST=${1:?manifest required}
TOK=${TOK:-experiments/_tokenizer_h1}
TASKS="ESOL Lipophilicity QM7 BBBP BACE Tox21 HIV"

aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors

RUNS=$($PY -c "
import json,os
m=json.load(open('$MANIFEST'))
print(' '.join(r['output_dir'] for r in m['runs']))
")
S3ROOT=$($PY -c "import json;print(json.load(open('$MANIFEST'))['s3_backup_root'])")

for d in $RUNS; do
    name=$(basename "$d")
    # Only evaluate what actually finished: verified.json is written solely on genuine completion,
    # so this cannot score a truncated encoder.
    if [ ! -f "$d/verified.json" ]; then
        echo "[h1cv] $name: not verified complete - skipping"; continue
    fi
    if [ -f "$d/moleculenet_cv/suite_summary.json" ] && \
       grep -q "HIV_nef1_MEAN" "$d/moleculenet_cv/suite_summary.json" 2>/dev/null; then
        echo "[h1cv] $name: CV already present - skipping"; continue
    fi
    if [ ! -f "$d/encoder/model.safetensors" ]; then
        echo "[h1cv] $name: no encoder - skipping"; continue
    fi
    echo "[h1cv] $name: 5-fold CV over $TASKS"
    $PY eval_v2.py --output_dir "$d/moleculenet_cv" --encoder "$d/encoder" --tokenizer "$TOK" \
        --pool mean --standardize zscore --head mlp --max_length 256 \
        --head_seeds 0 1 2 --cv_folds 5 --datasets $TASKS
    rc=$?
    # Upload each result as it lands rather than at the end: if the box dies or hits its deadline
    # mid-pass, the completed evaluations are already safe.
    aws s3 sync "$d/moleculenet_cv" "$S3ROOT/$name/moleculenet_cv" --only-show-errors
    echo "[h1cv] $name: rc=$rc"
done
echo "[h1cv] DONE"

#!/usr/bin/env bash
# 5-fold CV for the six H1 frac0p001 encoders, run as a STANDALONE eval pass (not a guard
# post-hook).
#
# Why standalone. phase2_worker.sh self-stops the box (`sudo shutdown`) the instant its completion
# gate is met, which races and pre-empts the guard's POST_HOOK -- so the CV that was supposed to
# run after training never did. The encoders are all safe in S3; only their CV is missing. This
# script needs no worker and no guard: it pulls each verified encoder from S3, CVs it, uploads,
# and stops the box itself when all six are done.
#
# Waits per-run: it CVs a run as soon as that run's verified.json appears in S3, so it starts on
# the already-finished encoders immediately and blocks only on the last one still training.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_h1
ROOT=experiments/climb_v2_h1
TOK=experiments/_tok_h1cv
TASKS="ESOL QM7 BBBP BACE Tox21 HIV"      # Lipophilicity excluded (blocklist predates it)
RUNS="scaling_canonical_frac0p001_s0 scaling_enumerated_frac0p001_s0
      scaling_canonical_frac0p001_s1 scaling_enumerated_frac0p001_s1
      scaling_canonical_frac0p001_s2 scaling_enumerated_frac0p001_s2"
MAX_WAIT_MIN=${MAX_WAIT_MIN:-120}

aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors
say(){ echo "[h1cvstd $(date -u +%H:%M:%S)] $*"; }

done_ok=0
for name in $RUNS; do
    # wait until this run is verified in S3 (the last encoder may still be training)
    waited=0
    until aws s3 ls "$S3/$name/verified.json" >/dev/null 2>&1; do
        [ "$waited" -ge "$MAX_WAIT_MIN" ] && { say "$name: not verified after ${MAX_WAIT_MIN}m - skipping"; break; }
        say "$name: waiting for verified.json (${waited}m)"; sleep 120; waited=$((waited+2))
    done
    aws s3 ls "$S3/$name/verified.json" >/dev/null 2>&1 || continue

    # idempotent: skip if CV with the per-cell rows already exists
    if aws s3 cp "$S3/$name/moleculenet_cv/moleculenet_summary.csv" - 2>/dev/null | grep -q "_cell"; then
        say "$name: CV already present - skipping"; done_ok=$((done_ok+1)); continue
    fi

    d="$ROOT/$name"; mkdir -p "$d"
    aws s3 sync "$S3/$name/encoder" "$d/encoder" --only-show-errors
    if [ ! -f "$d/encoder/model.safetensors" ]; then say "$name: NO ENCODER in S3 - skip"; continue; fi

    say "$name: 5-fold CV"
    $PY eval_v2.py --output_dir "$d/moleculenet_cv" --encoder "$d/encoder" --tokenizer "$TOK" \
        --pool mean --standardize zscore --head mlp --max_length 256 \
        --head_seeds 0 1 2 --cv_folds 5 --subsample_seed 0 --datasets $TASKS
    rc=$?
    aws s3 sync "$d/moleculenet_cv" "$S3/$name/moleculenet_cv" --only-show-errors
    say "$name: rc=$rc"; [ "$rc" -eq 0 ] && done_ok=$((done_ok+1))
done

for f in h1cvstd.log; do [ -f "$f" ] && aws s3 cp "$f" "$S3/_logs/$f" --only-show-errors; done
bash scripts/notify.sh "$([ "$done_ok" -ge 6 ] && echo DONE || echo ALERT)" \
    "H1 frac0p001 CV standalone: $done_ok/6" \
    "5-fold CV for the H1 frac0p001 encoders. Box terminating."
say "H1CVSTD_DONE $done_ok/6"
sudo shutdown -h now

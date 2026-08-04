#!/usr/bin/env bash
# SI vocab-size scaling law (wave climb_v2_vocab), all on ONE box.
#
# BPE and Unigram, each at vocab {261, 1000, 10000, 100000}, MLM/unsup only, 2M FP, one seed.
# Runs are ordered FAST-FIRST and the middle of the curve is interleaved across families, so the
# most informative + lowest-risk points (1k/10k, both families) land first and a partial figure is
# interpretable within a few hours; the token-heavy (261) and OOM-prone (100k) endpoints come last.
#
# Each run does train -> 5-fold CV -> upload BEFORE the next starts, so results stream to S3 as
# they finish. CV is a standalone eval here, NOT a guard post-hook -- phase2_worker's self-stop
# races and skips post-hooks (that dropped a CV yesterday), so this driver owns the whole loop.
set -uo pipefail
cd /home/ec2-user/CLIMB
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/climb_v2_vocab
TOKS3=s3://climb-s3-bucket/tokenizers_vocab
ROOT=experiments/climb_v2_vocab
TOKROOT=experiments/_vocab_tok
TASKS="ESOL QM7 BBBP BACE Tox21 HIV"         # Lipophilicity excluded (blocklist predates it)
FP=${FP:-2000000}
say(){ echo "[vocab $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$ROOT"
# fast-first, family-interleaved
RUNS="bpe_10000 unigram_10000 bpe_1000 unigram_1000 bpe_261 unigram_261 bpe_100000 unigram_100000"

# ---- step 0: tokenizers (idempotent; uploads to S3, leaves local) ----
say "building tokenizers"
$PY scripts/build_vocab_tokenizers.py --sample 2000000 --out "$TOKROOT" --s3 "$TOKS3" || { say "TOKENIZER BUILD FAILED"; exit 1; }

# ---- template config ----
aws s3 cp "$S3/../climb_v2_phase2/unsup_2M/config.yaml" /tmp/tmpl.yaml >/dev/null 2>&1 || \
  aws s3 cp s3://climb-s3-bucket/experiments/climb_v2_phase2/unsup_2M/config.yaml /tmp/tmpl.yaml
say "template config fetched"

ok=0; total=0
for name in $RUNS; do
    total=$((total+1))
    fam=${name%_*}; vocab=${name##*_}
    d="$ROOT/$name"; enc="$d/encoder"
    # idempotent: skip a run that already has CV in S3
    if aws s3 cp "$S3/$name/moleculenet_cv/moleculenet_summary.csv" - 2>/dev/null | grep -q "_cell"; then
        say "$name: already complete in S3 - skipping"; ok=$((ok+1)); continue
    fi
    mkdir -p "$d"
    tok="$TOKROOT/$name"
    [ -f "$tok/tokenizer.json" ] || aws s3 sync "$TOKS3/$name" "$tok" --only-show-errors

    say "=== $name: writing config ==="
    $PY scripts/vocab_write_config.py --template /tmp/tmpl.yaml --run-id "$name" \
        --tokenizer "$tok" --vocab "$vocab" --fp "$FP" --out "$d/config.yaml" || { say "$name: config FAILED"; continue; }

    say "=== $name: pretrain (2M FP) ==="
    $PY pretrain_v2.py --run_dir "$d" --config "$d/config.yaml"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$enc/model.safetensors" ]; then
        say "$name: TRAIN FAILED rc=$rc - uploading partial + continuing"
        aws s3 sync "$d" "$S3/$name" --exclude "*/tokenizer/*" --only-show-errors; continue
    fi
    # completion proof from achieved FP (top-up in pretrain_v2 holds the budget)
    ach=$($PY -c "import json;print(json.loads(open('$d/metrics.jsonl').read().strip().splitlines()[-1]).get('forward_passes_seen',0))" 2>/dev/null || echo 0)
    if [ "$ach" -ge $((FP*98/100)) ]; then
        $PY -c "import json;json.dump({'run_id':'$name','budget_fp':$FP,'final_fp':$ach,'fraction':round($ach/$FP,4)},open('$d/verified.json','w'))"
        say "$name: verified ($ach/$FP FP)"
    else
        say "$name: WARNING only $ach/$FP FP - not marking verified"
    fi

    say "=== $name: 5-fold CV ==="
    $PY eval_v2.py --output_dir "$d/moleculenet_cv" --encoder "$enc" --tokenizer "$tok" \
        --pool mean --standardize zscore --head mlp --max_length 256 \
        --head_seeds 0 1 2 --cv_folds 5 --subsample_seed 0 --datasets $TASKS
    cvrc=$?

    aws s3 sync "$d" "$S3/$name" --exclude "*/tokenizer/*" --only-show-errors
    say "$name: DONE (train rc=$rc, cv rc=$cvrc) -> $S3/$name"
    [ "$cvrc" -eq 0 ] && ok=$((ok+1))
    bash scripts/notify.sh INFO "vocab wave: $name done ($ok/$total so far)" \
        "$name trained + CV'd and uploaded to $S3/$name."
done

for f in vocab_wave.log; do [ -f "$f" ] && aws s3 cp "$f" "$S3/_logs/$f" --only-show-errors; done
bash scripts/notify.sh "$([ "$ok" -ge 8 ] && echo DONE || echo ALERT)" \
    "vocab-size scaling wave: $ok/8 complete" \
    "BPE + Unigram x {261,1k,10k,100k}, 2M FP, CV on 6 tasks. Results in $S3. Box terminating."
say "VOCAB_WAVE_DONE $ok/8"
sudo shutdown -h now

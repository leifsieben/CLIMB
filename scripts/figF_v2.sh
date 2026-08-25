#!/usr/bin/env bash
# fig_F v2: the same grid again, emitting PER-FOLD values so the lift figure can pair.
#
# NOTHING HERE MAY CLOBBER v1. Every output carries a _V2 stem and every upload goes to
# experiments/_figF_v2/. The v1 prefix experiments/_figF/ is READ-ONLY to this script, and gate 5
# re-counts it at the end to prove that stayed true. The Ames prediction directory is derived from
# the output stem inside concat_redundancy_panels.py, so a _V2 stem gives new directories rather
# than overwriting the six that fig_F's caption already rests on. Encoders and the npz tables are
# DOWNLOADED only; this script never writes to a checkpoint prefix.
#
# WORK QUEUE, NOT A STATIC SPLIT. Mordred blocks cost ~2x RDKit ones and the per-tag runs differ
# again, so any hand-assigned split idles a worker. Workers pop the next unclaimed cell with mkdir
# as the atomic lock, which self-balances and makes a dead worker cost one cell rather than a queue.
set -u
cd /home/ec2-user/CLIMB
NW=${NW:-4}
LOG=analysis/figF_v2.log
S3V1=s3://climb-s3-bucket/experiments/_figF
S3=s3://climb-s3-bucket/experiments/_figF_v2
PY=~/venvs/climb/bin/python
Q=analysis/_v2_queue; LK=analysis/_v2_locks
mkdir -p analysis/rigor "$LK"
say () { echo "[v2] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
die () { say "FATAL $*"; exit 2; }

# ---------------------------------------------------------------- stage + preflight
say "staging inputs (read-only pulls)"
aws s3 cp "$S3V1/_mordred_figF.npz"   figure_data/_mordred_figF.npz   --only-show-errors
aws s3 cp "$S3V1/_chemeleon_figF.npz" figure_data/_chemeleon_figF.npz --only-show-errors
aws s3 cp "$S3V1/_figF_smiles.json"   figure_data/_figF_smiles.json   --only-show-errors
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_mordred_figF.npz ]   || die "mordred npz absent"
[ -s figure_data/_chemeleon_figF.npz ] || die "chemeleon npz absent"
[ -s figure_data/_tokenizer/tokenizer.json ] || die "tokenizer absent"
for e in unsup_8M skip_dense_8M; do
  d=figure_data/climb_v2_phase2/$e/encoder
  [ -s "$d/model.safetensors" ] || aws s3 cp "s3://climb-s3-bucket/experiments/climb_v2_phase2/$e/encoder" "$d" --recursive --only-show-errors
  [ -s "$d/model.safetensors" ] || die "encoder $e absent after staging"
done
# LOAD-TEST, do not merely stat. A present-but-unloadable input and a working one are
# indistinguishable until the traceback, which last time cost four cells and ten seconds each.
$PY - <<'PYEOF' >> "$LOG" 2>&1 || die "preflight load test failed"
import json, numpy as np
from transformers import PreTrainedTokenizerFast, ModernBertModel
t = PreTrainedTokenizerFast.from_pretrained("figure_data/_tokenizer"); assert t.vocab_size > 0
for e in ("unsup_8M", "skip_dense_8M"):
    m = ModernBertModel.from_pretrained(f"figure_data/climb_v2_phase2/{e}/encoder",
                                        attn_implementation="sdpa", reference_compile=False)
    print(f"[preflight] {e} hidden={m.config.hidden_size}", flush=True)
want = set(json.load(open("figure_data/_figF_smiles.json"))["_all_unique"])
for f in ("_mordred_figF", "_chemeleon_figF"):
    z = np.load(f"figure_data/{f}.npz", allow_pickle=True)
    S, X = z["smiles"], z["X"]                 # hoist: npz members decode lazily
    have = {str(s) for s in S}
    miss = [s for s in want if s not in have]
    print(f"[preflight] {f}: {X.shape}, {len(miss)} of {len(want)} fig_F molecules missing", flush=True)
    assert not miss, f"{f} does not cover the fig_F molecules"
PYEOF
say "preflight OK -- encoders load, both tables cover every fig_F molecule"

# ---------------------------------------------------------------- build the queue
# SHARED cells carry TAG=SHARED and EMB=chemeleon: no block name contains "SHARED", so the patched
# feature_sets returns before touching any embedding, and EMB=chemeleon skips the encoder load at
# import. fp/desc/fp+desc are identical across tags (48 cells, 0 disagreements), so computing them
# once per family instead of three times is the whole saving.
: > "$Q"
for fam in rdkit mordred; do
  f=$fam; [ "$fam" = rdkit ] && f=rdkit_sameenv
  echo "molnet|$fam|SHARED|chemeleon|NONE|fp,desc,fp+desc|concat_${f}_SHARED_V2.csv" >> "$Q"
  echo "panels|$fam|SHARED|chemeleon|NONE|fp,desc,fp+desc|concat_panels_${f}_SHARED_V2.csv" >> "$Q"
  for spec in "CLMunsup:climb:figure_data/climb_v2_phase2/unsup_8M/encoder" \
              "CLMsup:climb:figure_data/climb_v2_phase2/skip_dense_8M/encoder" \
              "CheMel:chemeleon:figure_data/_chemeleon_figF.npz"; do
    tag=${spec%%:*}; rest=${spec#*:}; emb=${rest%%:*}; src=${rest#*:}
    B="$tag,fp+$tag,desc+$tag,fp+desc+$tag"
    echo "molnet|$fam|$tag|$emb|$src|$B|concat_${f}_${tag}_V2.csv" >> "$Q"
    echo "panels|$fam|$tag|$emb|$src|$B|concat_panels_${f}_${tag}_V2.csv" >> "$Q"
  done
done
NCELL=$(wc -l < "$Q"); say "queue built: $NCELL cells, $NW workers"

# ---------------------------------------------------------------- worker
worker () {
  local wid=$1 i=0 line
  while IFS= read -r line; do
    i=$((i+1))
    mkdir "$LK/$i" 2>/dev/null || continue          # atomic claim; taken -> next cell
    local sc dk tag emb src blocks out
    IFS='|' read -r sc dk tag emb src blocks out <<< "$line"
    if [ -s "analysis/rigor/$out" ] && [ -s "analysis/rigor/${out%.csv}_folds.csv" ]; then
      say "w$wid SKIP $out"; continue
    fi
    say "w$wid RUN $out [$blocks]"
    local kv=(); [ "$emb" = climb ] && kv=(CONCAT_ENC="$src")
    [ "$emb" = chemeleon ] && [ "$src" != NONE ] && kv=(CONCAT_FEATURES_NPZ="$src")
    if [ "$sc" = molnet ]; then
      env CONCAT_DESC="$dk" CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_OUT="$out" \
          CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ=figure_data/_mordred_figF.npz \
          OMP_NUM_THREADS=8 "${kv[@]}" $PY scripts/concat_redundancy.py >> "analysis/_v2_w$wid.log" 2>&1
    else
      env CONCAT_DESC="$dk" CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_PANEL_OUT="$out" \
          CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ=figure_data/_mordred_figF.npz \
          CONCAT_PANELS="MoleculeACE Ames" OMP_NUM_THREADS=8 "${kv[@]}" \
          $PY scripts/concat_redundancy_panels.py >> "analysis/_v2_w$wid.log" 2>&1
    fi
    local rc=$? n
    # COMPLETION IS COUNTED PER-FOLD ROWS, not a file. want=1 on a Polaris panel is how finished
    # work got withheld twice; the fold file is the deliverable now, so count THAT.
    n=$($PY -c "
import csv
try: print(sum(1 for _ in csv.DictReader(open('analysis/rigor/${out%.csv}_folds.csv'))))
except Exception: print(0)" 2>/dev/null)
    if [ "${n:-0}" -ge 1 ]; then say "w$wid DONE $out ($n fold rows, rc=$rc)"
    else say "w$wid INCOMPLETE $out (0 fold rows, rc=$rc)"; fi
  done < "$Q"
  say "w$wid drained"
}
for w in $(seq 1 "$NW"); do worker "$w" & done
wait
say "ALL WORKERS DRAINED"

# ---------------------------------------------------------------- upload (tables + folds + Ames)
for f in analysis/rigor/*_V2.csv analysis/rigor/*_V2_folds.csv; do
  [ -s "$f" ] && aws s3 cp "$f" "$S3/$(basename "$f")" --only-show-errors
done
# THE PREDICTIONS ARE THE AMES DELIVERABLE. Polaris withholds test labels, so no results row is
# ever written for Ames and a results-only sweep sees nothing missing. Not uploading these is
# exactly what cost 14 Ames cells when the last box terminated.
np=0
for d in figure_data/chemeleon_suite/polaris/*_V2; do
  [ -s "$d/test_predictions.csv" ] || continue
  aws s3 cp "$d/test_predictions.csv" "$S3/ames/$(basename "$d")/test_predictions.csv" --only-show-errors
  np=$((np+1))
done
say "uploaded tables, fold files, and $np Ames prediction sets"
for l in analysis/figF_v2.log analysis/_v2_w*.log; do
  [ -s "$l" ] && aws s3 cp "$l" "$S3/logs/$(basename "$l")" --only-show-errors
done

# ---------------------------------------------------------------- gates
g () { say "gate $1 OK -- $2"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/logs/figF_v2.log" --only-show-errors; exit 1; }

for f in analysis/rigor/*_V2.csv analysis/rigor/*_V2_folds.csv; do
  b=$(basename "$f"); loc=$(stat -c %s "$f")
  rem=$(aws s3api head-object --bucket climb-s3-bucket --key "experiments/_figF_v2/$b" --query ContentLength --output text 2>/dev/null)
  [ "$rem" = "$loc" ] || abort "$b is $loc bytes locally, $rem on S3"
done
g 1 "every V2 table and fold file byte-matched on S3"

nt=$(aws s3 ls "$S3/" | grep -c "_V2\.csv$"); nf=$(aws s3 ls "$S3/" | grep -c "_V2_folds\.csv$")
[ "$nt" -eq "$NCELL" ] && [ "$nf" -eq "$NCELL" ] || abort "expected $NCELL tables and $NCELL fold files, counted $nt and $nf"
g 2 "$nt tables and $nf fold files, one per queued cell"

na=$(aws s3 ls "$S3/ames/" --recursive | grep -c "test_predictions.csv$")
[ "$na" -eq 8 ] || abort "expected 8 Ames prediction sets (2 shared + 6 per-tag), counted $na"
g 3 "8 Ames prediction sets present -- the check that was missing last time"

orphan=""
for f in analysis/rigor/*_V2*.csv; do
  b=$(basename "$f"); aws s3 ls "$S3/$b" >/dev/null 2>&1 || orphan="$orphan $b"
done
[ -z "$orphan" ] || abort "single-copy on this box:$orphan"
g 4 "no single-copy results"

# V1 MUST BE UNTOUCHED. Asserting the counts is the only way to prove this script kept its promise.
v1t=$(aws s3 ls "$S3V1/" | grep -c "concat_.*\.csv$"); v1a=$(aws s3 ls "$S3V1/ames/" --recursive | grep -c "test_predictions.csv$")
[ "$v1t" -eq 24 ] && [ "$v1a" -eq 6 ] || abort "v1 changed: expected 24 tables and 6 ames, found $v1t and $v1a"
g 5 "v1 intact -- 24 tables, 6 Ames prediction sets, untouched"

say "ALL GATES PASSED -- terminating"
aws s3 cp "$LOG" "$S3/logs/figF_v2.log" --only-show-errors
sudo shutdown -h now

#!/usr/bin/env bash
# Stage 2: the twelve fig_F tables, DRAWN BLOCKS FIRST.
#
# Leif's layout has four ticks per panel -- RDKit | Mordred | ECFP4 | RDKit+ECFP4 -- each with and
# without the embedding. That is 6 of the 7 blocks from the RDKit family and only 2 from the
# Mordred family (Mordred's fp+desc combination is not drawn). Each block costs 5 XGBoost fits per
# dataset, so PASS A fits only what the figure draws and ships; PASS B fills in the surplus
# afterwards, into *_EXTRA.csv so it cannot clobber what the figure is already reading.
#
# SCHEDULING. The two CLIMB arms need no CheMeleon vectors, so they run first while the chemprop
# venv builds and the missing panel vectors are computed in parallel. CheMeleon runs last.
#
# CBS is deliberately absent: fig_F does not panel it, so Mordred on CBS is not worth the spend.
#
# BLOCK NAMES STAY `desc` IN BOTH FAMILIES. The figures session renames Mordred's to `mdesc` at
# READ time and asserts on a pre-namespaced table; namespacing here would break their verified
# contract. One uniform input, one rename, one owner.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis analysis/rigor
LOG=analysis/figF_stage2.log
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/_figF
NPZ=figure_data/_mordred_figF.npz
say () { echo "[figF-s2] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

[ -s "$NPZ" ] || { say "FATAL mordred table $NPZ absent"; exit 2; }
# ASSERT COVERAGE BEFORE SPENDING. The lookup is strict; without this it would raise partway
# through MoleculeACE, after the cheap panels were already paid for.
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL mordred table does not cover the fig_F molecules"; exit 2; }
import json, numpy as np, sys
z = np.load("figure_data/_mordred_figF.npz", allow_pickle=True)
S = z["smiles"]                       # hoist: npz members decode lazily
have = {str(s) for s in S}
d = json.load(open("figure_data/_figF_smiles.json"))["_all_unique"]
miss = [s for s in d if s not in have]
print(f"[preflight] mordred table {len(have)} molecules, {len(miss)} of {len(d)} fig_F missing")
sys.exit(1 if miss else 0)
PYEOF
say "preflight OK -- table covers every fig_F molecule"

# STAGE AND ASSERT EVERY INPUT. The first launch of this script assumed the encoders were already
# on the box and they were not: all four CLIMB cells died in from_pretrained inside ten seconds.
# That is the same "asserted everything except one input" failure that cost the xgb job twelve
# cells on a missing tokenizer earlier today. Assert each one, because a missing input and a
# working run are otherwise indistinguishable until the traceback.
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "FATAL tokenizer.json absent"; exit 2; }
$PY -c "
from transformers import PreTrainedTokenizerFast
t = PreTrainedTokenizerFast.from_pretrained('figure_data/_tokenizer')
assert t.vocab_size > 0
print(f'[preflight] tokenizer OK, vocab={t.vocab_size}')" >> "$LOG" 2>&1 \
  || { say "FATAL tokenizer will not load"; exit 2; }
for e in unsup_8M skip_dense_8M; do
  d=figure_data/climb_v2_phase2/$e/encoder
  [ -s "$d/model.safetensors" ] || aws s3 cp "s3://climb-s3-bucket/experiments/climb_v2_phase2/$e/encoder" "$d" --recursive --only-show-errors
  [ -s "$d/model.safetensors" ] || { say "FATAL encoder $e absent after staging"; exit 2; }
  $PY -c "
from transformers import ModernBertModel
m = ModernBertModel.from_pretrained('$d', attn_implementation='sdpa', reference_compile=False)
print('[preflight] encoder $e loads, hidden', m.config.hidden_size)" >> "$LOG" 2>&1 \
    || { say "FATAL encoder $e will not load"; exit 2; }
done
say "encoders and tokenizer staged and load-tested"

# blocks, per family, for a given tag
drawn_rdkit  () { echo "fp,desc,fp+desc,fp+$1,desc+$1,fp+desc+$1"; }
extra_rdkit  () { echo "$1"; }
drawn_mordred() { echo "desc,desc+$1"; }
extra_mordred() { echo "fp,fp+desc,$1,fp+$1,fp+desc+$1"; }

cell () {   # $1 script(molnet|panels) $2 desc $3 tag $4 emb $5 envk $6 envv $7 blocks $8 suffix
  local sc=$1 dk=$2 tag=$3 emb=$4 k=$5 v=$6 blocks=$7 sfx=$8
  local fam=$dk; [ "$dk" = "rdkit" ] && fam="rdkit_sameenv"
  local base out want script
  if [ "$sc" = "molnet" ]; then
    out="concat_${fam}_${tag}${sfx}.csv"; want=6; script=scripts/concat_redundancy.py
  else
    out="concat_panels_${fam}_${tag}${sfx}.csv"; want=1; script=scripts/concat_redundancy_panels.py
  fi
  if [ -s "analysis/rigor/$out" ]; then say "SKIP $out"; return 0; fi
  say "RUN $out  [$blocks]"
  if [ "$sc" = "molnet" ]; then
    env CONCAT_DESC="$dk" CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_OUT="$out" \
        CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ="$NPZ" "$k=$v" \
        $PY "$script" >> "$LOG" 2>&1
  else
    env CONCAT_DESC="$dk" CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_PANEL_OUT="$out" \
        CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ="$NPZ" CONCAT_PANELS="MoleculeACE Ames" "$k=$v" \
        $PY "$script" >> "$LOG" 2>&1
  fi
  local rc=$? n
  # COMPLETION IS COUNTED TASKS, NOT A FILE -- the script writes its csv even for a partial run.
  n=$($PY -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/$out'))}))
except Exception: print(0)" 2>/dev/null)
  if [ "${n:-0}" -ge "$want" ]; then
    aws s3 cp "analysis/rigor/$out" "$S3/$out" --only-show-errors
    say "DONE $out ($n/$want tasks, rc=$rc, uploaded)"
  else
    say "INCOMPLETE $out ($n/$want, rc=$rc) -- NOT uploaded"
  fi
}

# ---------------------------------------------------------------- PASS A: what the figure draws
say "PASS A -- drawn blocks only, CLIMB arms first (no CheMeleon dependency)"
for spec in "CLMunsup:unsup_8M" "CLMsup:skip_dense_8M"; do
  tag=${spec%%:*}; enc=${spec##*:}
  cell molnet rdkit   "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(drawn_rdkit $tag)"   ""
  cell molnet mordred "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(drawn_mordred $tag)" ""
  cell panels rdkit   "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(drawn_rdkit $tag)"   ""
  cell panels mordred "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(drawn_mordred $tag)" ""
done
say "PASS A: CLIMB arms done -- waiting on CheMeleon table if it is not ready"

CHNPZ=figure_data/_chemeleon_figF.npz
for i in $(seq 1 240); do [ -s "$CHNPZ" ] && break; sleep 30; done
if [ -s "$CHNPZ" ]; then
  cell molnet rdkit   CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(drawn_rdkit CheMel)"   ""
  cell molnet mordred CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(drawn_mordred CheMel)" ""
  cell panels rdkit   CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(drawn_rdkit CheMel)"   ""
  cell panels mordred CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(drawn_mordred CheMel)" ""
else
  say "CheMeleon table never appeared -- PASS A incomplete for CheMel"
fi
say "PASS A COMPLETE"

# ---------------------------------------------------------------- PASS B: the surplus, for the record
say "PASS B -- surplus blocks into *_EXTRA.csv (never clobbers what the figure reads)"
for spec in "CLMunsup:unsup_8M" "CLMsup:skip_dense_8M"; do
  tag=${spec%%:*}; enc=${spec##*:}
  cell molnet rdkit   "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(extra_rdkit $tag)"   _EXTRA
  cell molnet mordred "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(extra_mordred $tag)" _EXTRA
  cell panels rdkit   "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(extra_rdkit $tag)"   _EXTRA
  cell panels mordred "$tag" climb CONCAT_ENC "figure_data/climb_v2_phase2/$enc/encoder" "$(extra_mordred $tag)" _EXTRA
done
if [ -s "$CHNPZ" ]; then
  cell molnet rdkit   CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(extra_rdkit CheMel)"   _EXTRA
  cell molnet mordred CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(extra_mordred CheMel)" _EXTRA
  cell panels rdkit   CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(extra_rdkit CheMel)"   _EXTRA
  cell panels mordred CheMel chemeleon CONCAT_FEATURES_NPZ "$CHNPZ" "$(extra_mordred CheMel)" _EXTRA
fi
say "PASS B COMPLETE -- staying UP for inspection"

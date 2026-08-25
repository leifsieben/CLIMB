#!/usr/bin/env bash
# MoLFormer-c3 only. Separate from wave1b because that script is already running and bash reads a
# script incrementally -- editing a live one is how you get a syntax error at line 40 mid-run.
#
# TOKENIZER: ibm/MoLFormer-XL-both-10pct at MAIN, deliberately NOT the 7b12d946c181 pin. The pin
# exists because that old revision is the one that loads under the PINNED venv's transformers
# 4.57.3. In this python3.12 / transformers 5.15.1 environment the relationship inverts: the old
# revision's config imports transformers.onnx, which 5.x removed. Same repo, opposite revision,
# because the constraint is the environment, not the checkpoint.
#
# NOTE ON THE NAME: "c3-1.1B" is the PRETRAINING DATA scale (1.1B molecules), not parameters --
# the model is 44M params, hidden 768, the MoLFormer-XL architecture. Recorded so nobody reads it
# as a billion-parameter arm.
set -u
cd /home/ec2-user/CLIMB
NEW=~/venvs/clm_new/bin/python
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_c3.log
say () { echo "[c3] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

if [ ! -s figure_data/_molformer_c3.npz ]; then
  say "extract MoLFormer-c3-1.1B (tokenizer: ibm main)"
  $NEW scripts/extract_clm_embeddings.py --hf_model DeepChem/MoLFormer-c3-1.1B \
       --tokenizer_from ibm/MoLFormer-XL-both-10pct \
       --smiles figure_data/_figA_smiles.json --out figure_data/_molformer_c3.npz \
       --batch_size 64 >> analysis/figA_extract_molformer_c3.log 2>&1
fi
[ -s figure_data/_molformer_c3.npz ] || { say "FATAL extraction produced nothing"; exit 2; }
say "npz ready ($(stat -c %s figure_data/_molformer_c3.npz) bytes)"

for track in moleculeace polaris; do
  d=figure_data/chemeleon_suite/$track/molformer_c3
  [ -s "$d/verified.json" ] && { say "SKIP $track"; continue; }
  say "RUN molformer_c3 on $track"
  $PY scripts/chemeleon_suite_run.py --track "$track" --featurizer npz \
      --encoder figure_data/_molformer_c3.npz --model molformer_c3 --head mlp --seeds 42 117 709 \
      >> "analysis/figA_molformer_c3_${track}.log" 2>&1
  rc=$?
  n=$($PY scripts/count_cell_tasks.py "$d" "$track" 2>/dev/null || echo 0)
  want=30; [ "$track" = polaris ] && want=28
  if [ "${n:-0}" -ge "$want" ]; then
    say "DONE molformer_c3/$track ($n/$want, rc=$rc)"
    aws s3 cp "$d" "$S3/$track/molformer_c3" --recursive --only-show-errors
  else
    say "INCOMPLETE molformer_c3/$track ($n/$want, rc=$rc)"
  fi
done
say "C3 COMPLETE"

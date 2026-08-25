#!/usr/bin/env bash
# wave 1b: the two arms that will not load in the pinned environment.
#
# Extraction runs in ~/venvs/clm_new (python3.12, transformers 5.15.1) because transformers 5.x
# requires python>=3.10 and the pinned venv is 3.9 -- that, not the pin alone, was the real
# blocker. The PROBE still runs in the pinned venv on the resulting npz, so these arms go through
# the same head as every other arm and nothing else can move.
set -u
cd /home/ec2-user/CLIMB
NEW=~/venvs/clm_new/bin/python
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_wave1b.log
say () { echo "[1b] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

SM=figure_data/_figA_smiles.json
[ -s "$SM" ] || { say "FATAL $SM absent"; exit 2; }

# --- extract ---------------------------------------------------------------
if [ ! -s figure_data/_molformer_c3.npz ]; then
  say "extract MoLFormer-c3-1.1B"
  $NEW scripts/extract_clm_embeddings.py --hf_model DeepChem/MoLFormer-c3-1.1B \
       --tokenizer_from ibm/MoLFormer-XL-both-10pct --tokenizer_revision 7b12d946c181 \
       --smiles "$SM" --out figure_data/_molformer_c3.npz --batch_size 64 \
       >> analysis/figA_extract_molformer_c3.log 2>&1 || say "MoLFormer-c3 extraction FAILED"
fi
if [ ! -s figure_data/_selfies_ted.npz ]; then
  say "extract selfies-ted"
  $NEW scripts/extract_clm_embeddings.py --hf_model ibm-research/materials.selfies-ted \
       --smiles "$SM" --out figure_data/_selfies_ted.npz --selfies --batch_size 64 \
       >> analysis/figA_extract_selfies_ted.log 2>&1 || say "selfies-ted extraction FAILED"
fi
for f in figure_data/_molformer_c3.npz figure_data/_selfies_ted.npz; do
  [ -s "$f" ] && say "have $f ($(stat -c %s "$f") bytes)" || say "MISSING $f"
done

# --- probe, in the PINNED venv ---------------------------------------------
for spec in "molformer_c3|figure_data/_molformer_c3.npz" \
            "selfies_ted|figure_data/_selfies_ted.npz"; do
  IFS='|' read -r model npz <<< "$spec"
  [ -s "$npz" ] || { say "SKIP $model (no npz)"; continue; }
  for track in moleculeace polaris; do
    d=figure_data/chemeleon_suite/$track/$model
    [ -s "$d/verified.json" ] && { say "SKIP $model/$track (verified)"; continue; }
    say "RUN $model on $track"
    $PY scripts/chemeleon_suite_run.py --track "$track" --featurizer npz --encoder "$npz" \
        --model "$model" --head mlp --seeds 42 117 709 \
        >> "analysis/figA_${model}_${track}.log" 2>&1
    rc=$?
    n=$($PY scripts/count_cell_tasks.py "$d" "$track" 2>/dev/null || echo 0)
    want=30; [ "$track" = polaris ] && want=28
    if [ "${n:-0}" -ge "$want" ]; then
      say "DONE $model/$track ($n/$want, rc=$rc)"
      aws s3 cp "$d" "$S3/$track/$model" --recursive --only-show-errors
    else
      say "INCOMPLETE $model/$track ($n/$want, rc=$rc)"
    fi
  done
done
say "WAVE 1B COMPLETE"

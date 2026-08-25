#!/usr/bin/env bash
# fig_A wave 1: the two literature CLMs that load in the pinned environment, on both suite tracks.
#
# SEED AXIS. These have ONE released checkpoint each, so the replicate axis is the HEAD seed --
# the CheMeleon convention, verified from chemeleon_frozen's own verified.json ([42,117,709]), not
# assumed. The CLIMB arms vary the PRETRAINING seed instead with heads pinned; mixing the two
# conventions is a mistake this project has already made once.
#
# SMI-TED IS DELIBERATELY ABSENT. It needs pytorch-fast-transformers in a separate venv and is one
# arm of three; letting it gate the other two would cost hours for no reason. It joins as wave 1b.
#
# NOTHING HERE OVERWRITES: each arm writes figure_data/chemeleon_suite/<track>/<model>/ under its
# own model name, and uploads to experiments/figA_clms/. No existing prefix is written.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_wave1.log
mkdir -p analysis
say () { echo "[figA] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

say "wave 1 start"
for spec in "chemberta_mtr|DeepChem/ChemBERTa-77M-MTR|" \
            "molformer|ibm/MoLFormer-XL-both-10pct|7b12d946c181"; do
  IFS='|' read -r model hfid rev <<< "$spec"
  for track in moleculeace polaris; do
    d=figure_data/chemeleon_suite/$track/$model
    if [ -s "$d/verified.json" ]; then say "SKIP $model/$track (verified)"; continue; fi
    say "RUN $model on $track"
    revarg=(); [ -n "$rev" ] && revarg=(--hf_revision "$rev")
    $PY scripts/chemeleon_suite_run.py --track "$track" --featurizer hf_encoder \
        --hf_model "$hfid" "${revarg[@]}" --model "$model" --head mlp --seeds 42 117 709 \
        >> "analysis/figA_${model}_${track}.log" 2>&1
    rc=$?
    # COMPLETION IS COUNTED TASKS. A results.csv exists after a partial run too, and on the polaris
    # track it is header-only BY DESIGN because the labels are withheld -- counting rows there is
    # the trap that has withheld finished work three times on this project.
    n=$($PY scripts/count_cell_tasks.py "$d" "$track" 2>/dev/null || echo 0)
    want=30; [ "$track" = polaris ] && want=28
    if [ "${n:-0}" -ge "$want" ]; then
      say "DONE $model/$track ($n/$want tasks, rc=$rc)"
      aws s3 cp "$d" "$S3/$track/$model" --recursive --only-show-errors
    else
      say "INCOMPLETE $model/$track ($n/$want, rc=$rc)"
    fi
  done
done
say "WAVE 1 COMPLETE -- staying up for wave 1b (smi-ted) and Wong/FartDB"

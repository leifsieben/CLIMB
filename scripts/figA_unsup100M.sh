#!/usr/bin/env bash
# The one arm missing from Wong and FartDB: unsup_100M, added to fig_A after the 13-arm list was
# specced. ONE directory with the base head-seed triple (single pretraining run), per the convention
# in figA_wave3.sh's header.
#
# Deliberately NOT figA_wave3.sh: that script's skip test reads a LOCAL verified.json, and on a
# fresh box no local marker exists for the 39 cells that are already done on S3 -- it would rerun
# every one of them. Run the two datasets for the one arm directly.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_unsup100M.log
mkdir -p analysis
say () { echo "[100M] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

ENC=figure_data/climb_v2_phase2/unsup_100M/encoder
if [ ! -s "$ENC/model.safetensors" ]; then
  aws s3 cp s3://climb-s3-bucket/experiments/climb_v2_phase2/unsup_100M/encoder "$ENC" --recursive --only-show-errors
fi
[ -s "$ENC/model.safetensors" ] || { say "MISSING ENCODER -- refusing to produce a cell"; exit 1; }
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "MISSING TOKENIZER"; exit 1; }
say "encoder and tokenizer staged"

rc=0
for ds in wong fartdb; do
  if [ "$ds" = wong ]; then
    out=figure_data/wong_saureus/unsup_100M; script=scripts/wong_run.py; rem="$S3/wong/unsup_100M"
  else
    out=figure_data/fartdb/unsup_100M;       script=scripts/fartdb_multiclass.py; rem="$S3/fartdb/unsup_100M"
  fi
  # Do not overwrite a cell that already exists ON S3 -- the durable copy is the one that counts.
  if aws s3 ls "$rem/verified.json" >/dev/null 2>&1; then say "SKIP $ds -- already on S3"; continue; fi
  say "RUN $ds"
  $PY "$script" --model unsup_100M --featurizer encoder --encoder "$ENC" --tokenizer figure_data/_tokenizer \
      --head mlp --seeds 42 117 709 >> "analysis/figA_100M_${ds}.log" 2>&1
  if [ -s "$out/verified.json" ]; then
    aws s3 cp "$out" "$rem" --recursive --only-show-errors
    aws s3 ls "$rem/verified.json" >/dev/null 2>&1 && say "DONE $ds -> $rem" || { say "UPLOAD FAILED $ds"; rc=1; }
  else
    say "FAILED $ds (see analysis/figA_100M_${ds}.log)"; rc=1
  fi
done
aws s3 cp "$LOG" "$S3/logs/figA_unsup100M.log" --only-show-errors
say "finished rc=$rc"
exit $rc

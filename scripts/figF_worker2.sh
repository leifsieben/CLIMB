#!/usr/bin/env bash
# Second worker: the CheMeleon quarter of PASS A, run concurrently with the main driver.
#
# WHY THIS IS SAFE. The main driver works CLMunsup -> CLMsup -> CheMel in a fixed order and skips
# any table whose csv already exists. This worker takes CheMel, which the driver reaches LAST --
# hours from now -- so by the time it gets there these files exist and are skipped. The only way
# to collide is for both to start the SAME table at the same moment, which that ordering prevents.
# Load is ~16 of 32 cores with one driver, so this is otherwise idle capacity.
set -u
cd /home/ec2-user/CLIMB
LOG=analysis/figF_worker2.log
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/_figF
NPZ=figure_data/_mordred_figF.npz
CH=figure_data/_chemeleon_figF.npz
say () { echo "[figF-w2] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
[ -s "$CH" ] || { say "FATAL chemeleon table absent"; exit 2; }

cell () {   # $1 molnet|panels  $2 rdkit|mordred  $3 blocks
  local sc=$1 dk=$2 blocks=$3
  local fam=$dk; [ "$dk" = "rdkit" ] && fam="rdkit_sameenv"
  local out want script
  if [ "$sc" = "molnet" ]; then out="concat_${fam}_CheMel.csv"; want=6; script=scripts/concat_redundancy.py
  else out="concat_panels_${fam}_CheMel.csv"; want=1; script=scripts/concat_redundancy_panels.py; fi
  [ -s "analysis/rigor/$out" ] && { say "SKIP $out"; return 0; }
  say "RUN $out [$blocks]"
  if [ "$sc" = "molnet" ]; then
    env CONCAT_DESC="$dk" CONCAT_EMB=chemeleon CONCAT_TAG=CheMel CONCAT_OUT="$out" \
        CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ="$NPZ" CONCAT_FEATURES_NPZ="$CH" \
        $PY "$script" >> "$LOG" 2>&1
  else
    env CONCAT_DESC="$dk" CONCAT_EMB=chemeleon CONCAT_TAG=CheMel CONCAT_PANEL_OUT="$out" \
        CONCAT_BLOCKS="$blocks" CONCAT_MORDRED_NPZ="$NPZ" CONCAT_FEATURES_NPZ="$CH" \
        CONCAT_PANELS="MoleculeACE Ames" $PY "$script" >> "$LOG" 2>&1
  fi
  local rc=$? n
  n=$($PY -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/$out'))}))
except Exception: print(0)" 2>/dev/null)
  if [ "${n:-0}" -ge "$want" ]; then
    aws s3 cp "analysis/rigor/$out" "$S3/$out" --only-show-errors
    say "DONE $out ($n/$want, rc=$rc, uploaded)"
  else say "INCOMPLETE $out ($n/$want, rc=$rc)"; fi
}

cell molnet mordred "desc,desc+CheMel"
cell panels mordred "desc,desc+CheMel"
cell molnet rdkit   "fp,desc,fp+desc,fp+CheMel,desc+CheMel,fp+desc+CheMel"
cell panels rdkit   "fp,desc,fp+desc,fp+CheMel,desc+CheMel,fp+desc+CheMel"
say "WORKER2 COMPLETE"

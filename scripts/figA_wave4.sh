#!/usr/bin/env bash
# wave 4: MolNet (6) and CBS for the three literature arms x three replicate dirs.
#
# This closes the last coverage gap. The ten existing arms already have MolNet and CBS locally and
# are deliberately not recomputed.
#
# HEAD-SEED TRIPLES ON THE MolNet/CBS TREE differ from the suite tree's [42,117,709] convention:
#     <arm>     head seeds 0,1,2      <arm>_s1  3,4,5      <arm>_s2  6,7,8
# taken from the fig_A spec. They are disjoint, which is what makes three dirs three replicates
# rather than three copies of one ensemble.
#
# MolNet runs through eval_v2 with --features_npz: the same precomputed tables the suite tracks
# used, so an arm's vectors are identical across every category it appears in. CBS runs through
# cbs_run.py because it needs the dataset's OWN provided folds -- replacing them with scaffold
# folds would make our CBS numbers incomparable with every CBS number already in the paper.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_wave4.log
MOLNET="ESOL QM7 BBBP BACE HIV Tox21"
say () { echo "[w4] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

LOCK=analysis/.figA_wave4.lock; exec 9>"$LOCK"
flock -n 9 || { say "another wave4 holds the lock -- refusing"; exit 0; }

n_ok=0; n_bad=0
for arm in chemberta_mtr molformer_c3 selfies_ted; do
  npz=figure_data/_${arm}.npz
  [ -s "$npz" ] || { say "MISSING $npz -- $arm skipped entirely"; n_bad=$((n_bad+1)); continue; }
  has=$($PY -c "
import numpy as np
print('yes' if 'meta' in np.load('$npz', allow_pickle=False).files else 'no')")
  [ "$has" = yes ] || { say "REFUSING $arm -- $npz carries no provenance"; n_bad=$((n_bad+1)); continue; }
  for spec in ":0 1 2" "_s1:3 4 5" "_s2:6 7 8"; do
    sfx=${spec%%:*}; hs=${spec#*:}
    label="${arm}${sfx}"

    # ---- MolNet ----
    d=figure_data/molnet/$label
    if [ -s "$d/suite_summary.json" ]; then
      say "SKIP molnet/$label"
    else
      say "RUN molnet/$label (head_seeds $hs)"
      $PY eval_v2.py --featurizer encoder --features_npz "$npz" --datasets $MOLNET \
          --head_seeds $hs --head mlp --cv_folds 5 --output_dir "$d" \
          >> "analysis/figA_w4_molnet_${label}.log" 2>&1
      # COMPLETION IS COUNTED DATASETS, not a file: eval_v2 writes its summary even for a run that
      # lost datasets partway.
      n=$($PY -c "
import json
try:
    s=json.load(open('$d/suite_summary.json'))
    print(len({k.rsplit('_',1)[0].split('_')[0] for k in s if k.endswith('_MEAN')}))
except Exception: print(0)" 2>/dev/null)
      if [ "${n:-0}" -ge 6 ]; then
        aws s3 cp "$d" "$S3/molnet/$label" --recursive --only-show-errors
        say "DONE molnet/$label ($n/6 datasets)"; n_ok=$((n_ok+1))
      else
        say "INCOMPLETE molnet/$label ($n/6) -- see analysis/figA_w4_molnet_${label}.log"; n_bad=$((n_bad+1))
      fi
    fi

    # ---- CBS ----
    c=figure_data/cbs/$label
    if [ -s "$c/verified.json" ]; then
      say "SKIP cbs/$label"
    else
      say "RUN cbs/$label (seeds $hs)"
      $PY scripts/cbs_run.py --model "$label" --featurizer npz --encoder "$npz" \
          --head mlp --seeds $hs >> "analysis/figA_w4_cbs_${label}.log" 2>&1
      if [ -s "$c/verified.json" ]; then
        aws s3 cp "$c" "$S3/cbs/$label" --recursive --only-show-errors
        say "DONE cbs/$label"; n_ok=$((n_ok+1))
      else
        say "FAILED cbs/$label -- see analysis/figA_w4_cbs_${label}.log"; n_bad=$((n_bad+1))
      fi
    fi
  done
done
say "WAVE 4 COMPLETE -- $n_ok cells done, $n_bad failed/missing"

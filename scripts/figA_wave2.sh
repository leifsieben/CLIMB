#!/usr/bin/env bash
# wave 2: the three literature arms x THREE replicate dirs on the two suite tracks.
#
# REPLICATE AXIS. These are single-checkpoint models, so the axis is the HEAD SEED on disjoint
# triples -- read off chemeleon_frozen's own dirs rather than assumed:
#     chemeleon_frozen [42,117,709]   _s1 [43,118,710]   _s2 [44,119,711]
# The CLIMB arms vary PRETRAINING instead with heads pinned; the two conventions are not
# interchangeable and this project has already mixed them up once.
#
# All three arms are featurized from npz tables produced in ONE environment, so no arm in this
# ranked group carries a different transformers version from its neighbours.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_wave2.log
say () { echo "[w2] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

for arm in chemberta_mtr molformer_c3 selfies_ted; do
  npz=figure_data/_${arm}.npz
  if [ ! -s "$npz" ]; then say "SKIP $arm -- $npz absent"; continue; fi
  # REFUSE AN UNPROVENANCED TABLE. A table with full coverage and no meta scores perfectly well and
  # produces a result nothing can trace to a checkpoint.
  ok=$($PY -c "
import numpy as np
z=np.load('$npz', allow_pickle=False)
print('yes' if 'meta' in z.files else 'no')")
  if [ "$ok" != "yes" ]; then say "SKIP $arm -- $npz has no meta blob yet"; continue; fi
  for spec in ":42 117 709" "_s1:43 118 710" "_s2:44 119 711"; do
    sfx=${spec%%:*}; seeds=${spec#*:}
    for track in moleculeace polaris; do
      d=figure_data/chemeleon_suite/$track/${arm}${sfx}
      [ -s "$d/verified.json" ] && { say "SKIP ${arm}${sfx}/$track"; continue; }
      say "RUN ${arm}${sfx} on $track (seeds $seeds)"
      $PY scripts/chemeleon_suite_run.py --track "$track" --featurizer npz --encoder "$npz" \
          --model "${arm}${sfx}" --head mlp --seeds $seeds \
          >> "analysis/figA_${arm}${sfx}_${track}.log" 2>&1
      rc=$?
      n=$($PY scripts/count_cell_tasks.py "$d" "$track" 2>/dev/null || echo 0)
      want=30; [ "$track" = polaris ] && want=28
      if [ "${n:-0}" -ge "$want" ]; then
        say "DONE ${arm}${sfx}/$track ($n/$want, rc=$rc)"
        aws s3 cp "$d" "$S3/$track/${arm}${sfx}" --recursive --only-show-errors
      else
        say "INCOMPLETE ${arm}${sfx}/$track ($n/$want, rc=$rc)"
      fi
    done
  done
done
say "WAVE 2 COMPLETE"

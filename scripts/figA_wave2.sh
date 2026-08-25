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

# ONE INSTANCE ONLY. Two copies of this script ran concurrently once -- I relaunched the chain
# without checking whether the previous wave was still alive -- and they raced on
# chemberta_mtr_s1/moleculeace: one was mid-RUN while the other saw the half-written directory,
# printed SKIP, and moved on. Neither uploaded it, and the directory no longer exists. The log
# reads "RUN ... SKIP" with no DONE and no error anywhere.
LOCK=analysis/.figA_wave2.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[w2] another wave2 holds $LOCK -- refusing to run a second copy" | tee -a "$LOG"
  exit 0
fi

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
      # SKIP ONLY IF IT CAME FROM THE TABLE WE ARE USING NOW. "verified.json exists" is the wrong
      # test and it silently preserved six stale base dirs: chemberta_mtr's base was built through
      # --featurizer hf_encoder under transformers 4.57.3 while its own _s1/_s2 came from the npz
      # under 5.15.1 -- an environment split WITHIN a single arm's replicate trio, which is the
      # fig_F failure at its worst. molformer_c3 and selfies_ted bases came from the earlier
      # 113,209-molecule tables. All three looked complete and verified.
      fresh=$($PY -c "
import json, numpy as np
try:
    v = json.load(open('$d/verified.json'))
    p = v.get('npz_provenance') or {}
    z = np.load('$npz', allow_pickle=False)
    m = json.loads(str(z['meta']))
    print('yes' if (p.get('hf_model') == m['hf_model']
                    and p.get('n_molecules') == m['n_molecules']) else 'no')
except Exception:
    print('no')")
      [ "$fresh" = "yes" ] && { say "SKIP ${arm}${sfx}/$track (matches current table)"; continue; }
      [ -s "$d/verified.json" ] && say "STALE ${arm}${sfx}/$track -- rebuilding from the current table"
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

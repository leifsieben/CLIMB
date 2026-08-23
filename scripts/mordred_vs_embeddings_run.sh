#!/usr/bin/env bash
# Mordred descriptors vs learned embeddings -- does any embedding add anything on top of Mordred?
#
# WHY MORDRED SPECIFICALLY. CheMeleon is a D-MPNN pretrained to regress exactly the 1,613 Mordred
# 2D descriptors. Against RDKit descriptors, "CheMeleon adds nothing" compares two different
# feature families and is only suggestive. Against MORDRED it is the sharp version of the claim:
# if CheMeleon adds nothing on top of its OWN pretraining target, it compressed that target rather
# than learning structure beyond it. Calculator(descriptors, ignore_3D=True) is 1613 descriptors --
# that set exactly, not an approximation of it.
#
# Three embeddings, so the answer is not about one model:
#   unsup_8M       CLIMB unsupervised
#   skip_dense_8M  CLIMB supervised (skip+dense)
#   chemeleon      CheMeleon frozen -- the one whose pretraining target this IS
#
# Seven feature blocks per dataset, all through the SAME XGBoost under the SAME scaffold folds, so
# the comparison is controlled: fp, desc, fp+desc, EMB, fp+EMB, desc+EMB, fp+desc+EMB. The four
# the question actually needs are desc, desc+EMB, fp+desc, fp+desc+EMB; the rest are the controls
# that let the table be re-cut without another run.
#
# RUNS IN THE REFERENCE ENVIRONMENT, ON PURPOSE. The Mordred table is keyed on the SMILES this
# machine's loaders produce, and the lookup is strict -- a box whose deepchem parses Tox21
# differently (7,831 vs 7,823 molecules) would RAISE rather than silently score a different
# molecule set. That is the intended behaviour, and it is why this stays local.
set -u
cd "$(dirname "$0")/.."
LOG=analysis/mordred_vs_emb.log
mkdir -p analysis/rigor
say () { echo "[mordred-emb] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

MORDRED_NPZ=figure_data/_mordred_features.npz
[ -s "$MORDRED_NPZ" ] || { say "FATAL $MORDRED_NPZ absent"; exit 2; }
# Assert the table covers every molecule BEFORE spending hours: the strict lookup would otherwise
# raise partway through HIV, after the cheap datasets had already been paid for.
python3 - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL Mordred table does not cover the MolNet sets"; exit 2; }
import json, numpy as np, sys
z = np.load("figure_data/_mordred_features.npz", allow_pickle=True)
S = z["smiles"]                      # hoist: npz members decode lazily
have = {str(s) for s in S}
d = json.load(open("figure_data/_molnet_smiles.json"))["_all_unique"]
miss = [s for s in d if s not in have]
print(f"[preflight] Mordred table {len(have)} molecules, {len(miss)} of {len(d)} MolNet missing")
sys.exit(1 if miss else 0)
PYEOF
say "preflight OK -- Mordred table covers every MolNet molecule"

run_model () {   # $1 tag  $2 emb  $3 extra-env-name  $4 extra-env-value
  local tag=$1 emb=$2 k=$3 v=$4
  local out="concat_mordred_${tag}.csv"
  if [ -s "analysis/rigor/$out" ]; then say "SKIP $tag (analysis/rigor/$out exists)"; return 0; fi
  say "RUN $tag (emb=$emb)"
  env CONCAT_DESC=mordred CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_OUT="$out" "$k=$v" \
      python3 scripts/concat_redundancy.py >> "$LOG" 2>&1
  local rc=$?
  # COMPLETION IS COUNTED DATASETS, NOT A FILE. The script writes its csv at the end, but a
  # partial write or a crashed dataset still leaves something on disk.
  local n
  n=$(python3 -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/$out'))}))
except Exception: print(0)" 2>/dev/null)
  if [ "${n:-0}" -ge 6 ]; then say "DONE $tag ($n/6 datasets, rc=$rc)"
  else say "INCOMPLETE $tag ($n/6 datasets, rc=$rc) -- left for inspection"; fi
}

run_model CLMunsup climb CONCAT_ENC figure_data/climb_v2_phase2/unsup_8M/encoder
run_model CLMsup   climb CONCAT_ENC figure_data/climb_v2_phase2/skip_dense_8M/encoder
run_model CheMel   chemeleon CONCAT_FEATURES_NPZ figure_data/_chemeleon_features.npz

say "all three models attempted"
for t in CLMunsup CLMsup CheMel; do
  n=$(python3 -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/concat_mordred_$t.csv'))}))
except Exception: print(0)" 2>/dev/null)
  say "  $t: $n/6 datasets"
done

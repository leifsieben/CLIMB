#!/usr/bin/env bash
# The SUITE half of fig_F's Mordred axis: MoleculeACE, Ames and CBS, for all three embeddings,
# with a same-environment RDKit arm beside it.
#
# WHY BOTH ARMS. fig_F's six canonical panels are MoleculeACE, HIV, BACE, Ames, Tox21, QM7. The
# MolNet run covers HIV/BACE/Tox21/QM7; without this, two of six panels would read "not run" and
# a partly-filled row looks like a hole rather than a result. And the RDKit arm is regenerated
# HERE rather than reused from the published tables for the same reason it was on MolNet: the
# published ones came from another environment, and plain `fp` -- identical code, no descriptors,
# no embedding -- already differs by up to 0.22 fold SD between environments. The figure's logic
# is "change exactly one thing"; descriptor family AND environment is two.
#
# BLOCK NAMING STAYS `desc` IN BOTH ARMS, deliberately. The figures session renames Mordred's block
# to `mdesc` at READ time, because both families otherwise key desc/fp+desc/desc+CLM identically
# and a drop_duplicates(keep="first") would silently discard an entire descriptor family with every
# bar still drawn. Namespacing here as well would give two mechanisms for one job and break their
# verified rename. One uniform input, one rename, one owner.
set -u
cd "$(dirname "$0")/.."
LOG=analysis/panels_mordred.log
mkdir -p analysis/rigor
say () { echo "[panels] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

NPZ=figure_data/_mordred_features.npz
[ -s "$NPZ" ] || { say "FATAL $NPZ absent"; exit 2; }
# ASSERT COVERAGE BEFORE SPENDING HOURS. The lookup is strict and would otherwise raise partway
# through MoleculeACE, after the cheap panels were already paid for.
python3 - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL Mordred table does not cover the panel molecules"; exit 2; }
import json, numpy as np, sys
z = np.load("figure_data/_mordred_features.npz", allow_pickle=True)
S = z["smiles"]                      # hoist: npz members decode lazily
have = {str(s) for s in S}
d = json.load(open("figure_data/_panel_smiles.json"))["_all_unique"]
miss = [s for s in d if s not in have]
print(f"[preflight] table {len(have)} molecules, {len(miss)} of {len(d)} panel molecules missing")
sys.exit(1 if miss else 0)
PYEOF
say "preflight OK -- table covers every panel molecule"

run () {   # $1 desc-kind  $2 tag  $3 emb  $4 env-name  $5 env-value
  local dk=$1 tag=$2 emb=$3 k=$4 v=$5
  local out="concat_panels_${dk}_${tag}.csv"
  [ "$dk" = "rdkit" ] && out="concat_panels_rdkit_sameenv_${tag}.csv"
  if [ -s "analysis/rigor/$out" ]; then say "SKIP $dk/$tag"; return 0; fi
  say "RUN $dk/$tag"
  env CONCAT_DESC="$dk" CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_PANEL_OUT="$out" "$k=$v" \
      python3 scripts/concat_redundancy_panels.py >> "$LOG" 2>&1
  local rc=$? n
  # COMPLETION IS COUNTED PANELS, NOT A FILE.
  n=$(python3 -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/$out'))}))
except Exception: print(0)" 2>/dev/null)
  [ "${n:-0}" -ge 3 ] && say "DONE $dk/$tag ($n/3 panels, rc=$rc)" || say "INCOMPLETE $dk/$tag ($n/3, rc=$rc)"
}

for dk in mordred rdkit; do
  run "$dk" CLMunsup climb     CONCAT_ENC figure_data/climb_v2_phase2/unsup_8M/encoder
  run "$dk" CLMsup   climb     CONCAT_ENC figure_data/climb_v2_phase2/skip_dense_8M/encoder
  run "$dk" CheMel   chemeleon CONCAT_FEATURES_NPZ figure_data/_chemeleon_features.npz
done
say "all six panel tables attempted"

#!/usr/bin/env bash
# Re-run the RDKit-descriptor arm of the concat experiment IN THIS ENVIRONMENT, for all three
# embeddings, so the figure's RDKit and Mordred columns are comparable.
#
# WHY. The published RDKit tables were produced elsewhere. The `fp` block -- identical code, no
# descriptors and no embedding in it -- differs between them and tonight's run by up to 0.22 fold
# SD (QM7 rmse 216.1573 vs 216.6266). That is small, but it is a pure environment term sitting
# inside any RDKit-vs-Mordred difference the figure would draw, and the figure's whole logic is
# "hold everything fixed, change one thing". Changing the descriptor family AND the environment
# at once is two things.
#
# Same script, same folds, same seeds, same machine, same hour -- only CONCAT_DESC differs.
set -u
cd "$(dirname "$0")/.."
LOG=analysis/rdkit_sameenv.log
mkdir -p analysis/rigor
say () { echo "[rdkit-sameenv] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

run_model () {   # $1 tag  $2 emb  $3 env-name  $4 env-value
  local tag=$1 emb=$2 k=$3 v=$4
  local out="concat_rdkit_sameenv_${tag}.csv"
  if [ -s "analysis/rigor/$out" ]; then say "SKIP $tag"; return 0; fi
  say "RUN $tag (emb=$emb, desc=rdkit)"
  env CONCAT_DESC=rdkit CONCAT_EMB="$emb" CONCAT_TAG="$tag" CONCAT_OUT="$out" "$k=$v" \
      python3 scripts/concat_redundancy.py >> "$LOG" 2>&1
  local rc=$? n
  n=$(python3 -c "
import csv
try: print(len({r['task'] for r in csv.DictReader(open('analysis/rigor/$out'))}))
except Exception: print(0)" 2>/dev/null)
  [ "${n:-0}" -ge 6 ] && say "DONE $tag ($n/6, rc=$rc)" || say "INCOMPLETE $tag ($n/6, rc=$rc)"
}

run_model CLMunsup climb CONCAT_ENC figure_data/climb_v2_phase2/unsup_8M/encoder
run_model CLMsup   climb CONCAT_ENC figure_data/climb_v2_phase2/skip_dense_8M/encoder
run_model CheMel   chemeleon CONCAT_FEATURES_NPZ figure_data/_chemeleon_features.npz
say "all three done"

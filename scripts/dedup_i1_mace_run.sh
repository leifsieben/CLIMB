#!/usr/bin/env bash
# fig_C1's x-axis for the canonical suite: TRUE max Tanimoto to the FULL 12-shard corpus for
# MoleculeACE's test molecules (plus the exact-match / near-dup counts).
#
# Why not corpus_similarity.csv: that scores against a 499,837-molecule SUBSAMPLE of one shard, so
# its values are LOWER BOUNDS. fig_C1 turns on identifying corpus-IDENTICAL molecules
# (Tanimoto == 1.0) to separate memorization from interpolation, and a ~4% sample cannot establish
# identity. This driver reads all 12 shards.
#
# Of the canonical six only MoleculeACE and QM7 are regression, so those two are fig_C1's canonical
# form; QM7 is already done, MoleculeACE is what is added here. CPU/RDKit, no GPU.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis/dedup_i1
LOG=analysis/dedup_i1_mace.log
say() { echo "[i1] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start"

mkdir -p figure_data/_tanimoto
[ -f figure_data/_tanimoto/corpus_similarity.csv ] || \
  aws s3 cp s3://climb-s3-bucket/experiments/_tanimoto/corpus_similarity.csv \
    figure_data/_tanimoto/corpus_similarity.csv --only-show-errors
n=$(awk -F, '$2=="MoleculeACE"' figure_data/_tanimoto/corpus_similarity.csv 2>/dev/null | wc -l)
say "MoleculeACE molecules available: $n"
[ "$n" -gt 1000 ] || { say "FATAL corpus_similarity.csv lacks MoleculeACE rows -> staying UP"; exit 1; }

# --mode full is REQUIRED: the default is "exact", which computes the exact-match/near-dup table
# for every dataset but never runs the full-12-shard Tanimoto pass, so
# full_corpus_similarity_i1.csv is never written. That is what the first attempt did.
I1_TASKS="QM7 MoleculeACE" ~/venvs/climb/bin/python scripts/dedup_i1_reanalysis.py --mode full >> "$LOG" 2>&1
rc=$?; say "rc=$rc"

OUT=analysis/dedup_i1/full_corpus_similarity_i1.csv
if [ -s "$OUT" ] && awk -F, '$2=="MoleculeACE"' "$OUT" | head -1 | grep -q .; then
  aws s3 cp --recursive analysis/dedup_i1 s3://climb-s3-bucket/experiments/analysis_rigor/dedup_i1 --only-show-errors
  say "COMPLETE ($(awk -F, '$2=="MoleculeACE"' "$OUT" | wc -l) MoleculeACE rows) -> shutdown"
  sudo shutdown -h now
else
  say "INCOMPLETE (no MoleculeACE rows in $OUT) -> staying UP"
fi

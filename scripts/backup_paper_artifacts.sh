#!/usr/bin/env bash
# Backup of everything the paper and its peer review depend on, into a SEPARATE bucket.
#
# Why a second bucket rather than relying on versioning: versioning protects against overwriting
# an object, but not against deleting the prefix, a bad recursive `aws s3 rm`, or a lifecycle rule
# quietly expiring old versions. The backup bucket has versioning on and NO expiry lifecycle, so
# nothing in it ages out.
#
# Scope is deliberate, not "everything": experiments/robust_matrix alone is 3.72 TB of an old wave
# that no figure reads, so copying the whole bucket would cost ~$86/month to protect data nobody
# needs. The paper-critical set is ~22 GB.
#
# COPY ONLY, never delete: no --delete flag anywhere. A file removed from the source stays in the
# backup, which is the entire point.
set -uo pipefail
SRC=s3://climb-s3-bucket
DST=s3://climb-paper-backup-075120018132
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

log(){ echo "[backup $(date -u +%H:%M:%S)] $*"; }

# Waves every figure is built from, plus the derived analysis products and the tokenizer/corpora
# needed to retrain or re-evaluate any of them from scratch.
# climb_v2_labeleff_v2_frac_e2e = the CURRENT label-efficiency e2e data (per-task fractions).
# climb_v2_labeleff / climb_v2_labeleff_v2 are SUPERSEDED (absolute budgets) — backed up for
# provenance only; the live B1p1 figure reads analysis/rigor/label_efficiency_fractions_*.csv.
WAVES="climb_v2_phase2 climb_v2_ablation_dedup climb_v2_ablation climb_v2 climb_v2_h1
       climb_v2_headline climb_v2_labeleff climb_v2_labeleff_v2 climb_v2_labeleff_v2_frac_e2e climb_v2_lrsweep"

for w in $WAVES; do
    log "experiments/$w"
    aws s3 sync "$SRC/experiments/$w" "$DST/experiments/$w" --only-show-errors
done

for p in derived tokenizer_10M configs; do
    log "$p"
    aws s3 sync "$SRC/$p" "$DST/$p" --only-show-errors
done

# The tokenized corpora are what makes a retrain reproducible; without them a reviewer cannot
# regenerate any encoder. ~8.7 GB.
log "tokenized_sources (corpora)"
aws s3 sync "$SRC/tokenized_sources" "$DST/tokenized_sources" \
    --exclude "pubchem_124m_full/*" --only-show-errors    # 124M download still in flight
log "tokenized (supervised parquet)"
aws s3 sync "$SRC/tokenized" "$DST/tokenized" --only-show-errors

# A manifest of what this backup run covered, so a later reader can tell what was in scope.
{
  echo "backup_utc=$STAMP"
  echo "source=$SRC"
  echo "waves=$WAVES"
  echo "excluded=experiments/robust_matrix (3.72TB, no figure reads it); tokenized_sources/pubchem_124m_full (in flight)"
} | aws s3 cp - "$DST/_backup_manifests/$STAMP.txt"

log "sizing the backup"
aws s3 ls "$DST" --recursive --summarize | tail -2
log "BACKUP_DONE $STAMP"

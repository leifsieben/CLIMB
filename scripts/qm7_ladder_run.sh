#!/usr/bin/env bash
# QM7 native units for the SCALING LADDER rungs (fig_B), and optionally other stale regression
# tasks. Same defect and same fix as the mainline: eval_v2 has scaled targets per fold and
# inverse-transformed before scoring for a month, so these stored z-scored values are artifacts
# predating that change. Without this, fig_B's QM7 rungs stay z-scored while fig_A's QM7 bars are
# native -- the same figure-to-figure unit split we just removed.
#
# RUNS / DATASETS / OUTSUB / SHARD / NSHARD come from the environment.
set -u
cd /home/ec2-user/CLIMB; mkdir -p analysis
LOG=analysis/qm7_ladder.log
say() { echo "[ladder] $(date -u +%FT%TZ) $*" >> "$LOG"; }
say "start shard ${SHARD:-0}/${NSHARD:-1} datasets=${DATASETS:-QM7}"
RUNS="${RUNS:-}" DATASETS="${DATASETS:-QM7}" OUTSUB="${OUTSUB:-moleculenet_cv_qm7native}" \
  SHARD="${SHARD:-0}" NSHARD="${NSHARD:-1}" \
  ~/venvs/climb/bin/python scripts/qm7_native_reeval.py >> "$LOG" 2>&1
say "rc=$?"
if [ -f "figure_data/QM7_NATIVE_DONE_${SHARD:-0}" ]; then say "COMPLETE -> shutdown"; sudo shutdown -h now
else say "INCOMPLETE -> staying UP for inspection"; fi

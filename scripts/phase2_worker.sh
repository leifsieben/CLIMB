#!/usr/bin/env bash
# Phase-2 per-worker runner. Runs ONE pre-split worker manifest, periodically syncs
# results to S3 (a sidecar loop, so a spot reclaim can't lose completed runs or the
# latest long-run checkpoint), then does a final sync and self-stops.
#
# Usage (on the box, detached):
#   nohup bash scripts/phase2_worker.sh <worker_manifest.json> <worker_name> &
set -x
cd /home/ec2-user/CLIMB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
PY=/home/ec2-user/venvs/climb/bin/python
MANIFEST="${1:?usage: phase2_worker.sh <manifest> <worker_name>}"
WORKER="${2:?usage: phase2_worker.sh <manifest> <worker_name>}"
LOCAL=experiments/climb_v2_phase2
S3=s3://climb-s3-bucket/experiments/climb_v2_phase2

# sidecar: push results (incl. periodic encoder checkpoints) to S3 every 10 min
( while true; do
    aws s3 sync "$LOCAL" "$S3" --exclude "*/tokenizer/*" >/dev/null 2>&1
    sleep 600
  done ) &
SIDECAR=$!

# shared descriptor stats (identical normalization across every box); harmless if absent
aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json >/dev/null 2>&1 || true

# encoders are needed as warm-start bases only in stage 2 (u2s); pull any that exist
aws s3 sync "$S3" "$LOCAL" --exclude "*/moleculenet/*" >/dev/null 2>&1 || true

$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --worker_name "$WORKER"

kill $SIDECAR 2>/dev/null || true
# final authoritative sync (include encoders — stage 2 warm-starts from them)
aws s3 sync "$LOCAL" "$S3" --exclude "*/tokenizer/*"
echo "PHASE2 WORKER $WORKER DONE $(date -u)"
sudo shutdown -h now

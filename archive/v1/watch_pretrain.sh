#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-}"
CLUSTER_CONFIG="${2:-}"
INTERVAL="${3:-300}"
SNAPSHOT_DIR="${4:-./monitoring}"

if [[ -z "$MANIFEST" || -z "$CLUSTER_CONFIG" ]]; then
  echo "Usage: $0 <manifest.json> <cluster_config.json> [interval_seconds] [snapshot_dir]" >&2
  exit 1
fi

mkdir -p "$SNAPSHOT_DIR"

trap 'echo "Stopped."; exit 0' INT TERM

while true; do
  clear
  echo "============================================================"
  echo "CLIMB Monitor $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "============================================================"
  python3 scripts/monitor_cluster.py \
    --manifest "$MANIFEST" \
    --cluster_config "$CLUSTER_CONFIG" \
    --output_json "$SNAPSHOT_DIR/cluster_snapshot.json" \
    --output_csv "$SNAPSHOT_DIR/cluster_snapshot.csv"
  echo
  echo "Next poll in ${INTERVAL}s"
  sleep "$INTERVAL"
done

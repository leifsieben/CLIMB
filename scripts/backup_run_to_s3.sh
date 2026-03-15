#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-}"
S3_DEST="${2:-}"

if [[ -z "$SRC_DIR" || -z "$S3_DEST" ]]; then
  echo "Usage: $0 <local_run_dir> <s3_dest>" >&2
  exit 1
fi

TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
STATUS_PATH="${SRC_DIR%/}/backup_status.json"

cat > "$STATUS_PATH" <<EOF
{"last_sync_utc":"$TS_UTC","s3_path":"$S3_DEST"}
EOF

aws s3 sync "$SRC_DIR" "$S3_DEST" --no-progress

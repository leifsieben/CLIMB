#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/backup_hpo_to_s3.sh <local_output_dir> <s3_uri>"
  exit 1
fi

local_output_dir="$1"
s3_uri="$2"

aws s3 sync "$local_output_dir" "$s3_uri" \
  --exclude "_trial_tmp/*" \
  --exclude "*.pt" \
  --exclude "*.bin" \
  --exclude "*.safetensors"

echo "Synced $local_output_dir to $s3_uri"

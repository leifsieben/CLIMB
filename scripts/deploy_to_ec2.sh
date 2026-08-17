#!/usr/bin/env bash
# Deploy ONLY the code the v2 pipeline needs to an EC2 worker.
#
# Ships the runtime Python modules, configs, scripts, tests, and the raw-SMILES prep
# helper — and nothing else. Explicitly excludes archive/ (legacy v1), notebooks,
# data files, virtualenvs, local experiment outputs, and the key file. So the EC2 box
# only ever sees the ~20 files that actually run the wave.
#
# Usage:
#   scripts/deploy_to_ec2.sh <host-ip> [remote_dir]
# Example:
#   scripts/deploy_to_ec2.sh 54.173.232.108
set -euo pipefail

HOST_IP="${1:?usage: deploy_to_ec2.sh <host-ip> [remote_dir]}"
REMOTE_DIR="${2:-/home/ec2-user/CLIMB}"
USER="${EC2_USER:-ec2-user}"
KEY="${EC2_KEY:-$(cd "$(dirname "$0")/.." && pwd)/climb-gpu-key.pem}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"
# Rule order matters: prune legacy/data/venv dirs FIRST, then include code files,
# then exclude everything else.
rsync -rlptz --prune-empty-dirs \
  --exclude='.git' --exclude='.claude' --exclude='archive' --exclude='climb' --exclude='.venv_sanity' \
  --exclude='experiments' --exclude='zinc_SMILES' --exclude='raw_data' \
  --exclude='canonical_sup' --exclude='local_prototyping_data' --exclude='hp_search_trial_0' \
  --exclude='preparing_datasets/raw_data' --exclude='preparing_datasets/archive' \
  --exclude='*.pem' \
  --include='*/' \
  --include='*.py' --include='*.yaml' --include='*.json' --include='*.sh' --include='*.txt' \
  --exclude='*' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20" \
  "$REPO_ROOT/" "$USER@$HOST_IP:$REMOTE_DIR/"

echo "Deployed clean code set to $USER@$HOST_IP:$REMOTE_DIR"

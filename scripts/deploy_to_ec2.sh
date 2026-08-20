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

# ---- DATA, NOT JUST CODE ------------------------------------------------------------------------
# This script ships code and, by the --include list above, only .py/.yaml/.json/.sh/.txt. Every .csv
# is excluded. That is the right default -- nobody wants a 223 MB parquet rsync'd to each box -- but
# it silently produced the SAME failure three times on 2026-08-20:
#
#   concat_redundancy_panels.py   chemeleon_suite/tasks/*.txt + data/cbs.csv   -> FileNotFoundError
#   cbs_e2e.py                    data/cbs.csv                                 -> died at fold 1
#   chemeleon_suite_run.py        chemeleon_suite/data/{moleculeace,polaris}/  -> would have
#
# In each case the job launched cleanly, ran for minutes, and died at the point of first USE, which
# is the most expensive place to find out. The files are small and live on S3 already, so pull them
# ON THE BOX rather than pushing from here.
#
# Opt out with DEPLOY_DATA=0 for a code-only push.
if [ "${DEPLOY_DATA:-1}" = "1" ]; then
  echo "Staging the small data set the runners assume..."
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$USER@$HOST_IP" "
    set -e
    cd '$REMOTE_DIR'
    mkdir -p data chemeleon_suite/data/moleculeace chemeleon_suite/data/polaris chemeleon_suite/tasks
    # The task manifests ARE .txt and so are in the rsync include list, but a box that was set up
    # by some other route will not have them -- which is how the fig_F panels pass died. Pull them
    # from S3 too, so this staging step can self-heal instead of only reporting the gap.
    aws s3 sync s3://climb-s3-bucket/datasets/tasks chemeleon_suite/tasks --only-show-errors
    [ -s data/cbs.csv ] || aws s3 cp s3://climb-s3-bucket/datasets/cbs.csv data/cbs.csv --only-show-errors
    aws s3 sync s3://climb-s3-bucket/datasets/moleculeace chemeleon_suite/data/moleculeace --only-show-errors
    aws s3 sync s3://climb-s3-bucket/datasets/polaris     chemeleon_suite/data/polaris     --only-show-errors
    # Assert HERE, where it is cheap, instead of at first use hours in.
    nc=\$(wc -l < data/cbs.csv 2>/dev/null || echo 0)
    nm=\$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)
    np=\$(ls chemeleon_suite/data/polaris/*.csv 2>/dev/null | wc -l)
    nt=\$(ls chemeleon_suite/tasks/*.txt 2>/dev/null | wc -l)
    echo \"  cbs.csv \$nc rows | moleculeace \$nm csv | polaris \$np csv | task manifests \$nt\"
    [ \"\$nc\" -gt 10000 ] && [ \"\$nm\" -ge 30 ] && [ \"\$np\" -ge 28 ] && [ \"\$nt\" -ge 2 ] || {
      echo '  DATA STAGING INCOMPLETE -- runners that read these will fail at first use'; exit 1; }
  " || { echo "Data staging FAILED on $HOST_IP -- fix before launching a job"; exit 1; }
  echo "Data staged and asserted on $HOST_IP"
fi

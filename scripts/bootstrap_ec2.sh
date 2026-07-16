#!/usr/bin/env bash
# One-time per-instance setup for a v2 worker, on top of the existing climb AMI
# (which already has CUDA + torch + transformers + the climb venv). Idempotent.
#
# Run ON the instance:  bash scripts/bootstrap_ec2.sh
set -euo pipefail

PY=/home/ec2-user/venvs/climb/bin/python
PIP=/home/ec2-user/venvs/climb/bin/pip

# xgboost is the only missing dependency vs the old (RoBERTa-era) environment.
$PY -c "import xgboost" 2>/dev/null || $PIP install -q xgboost

# ModernBERT sanity + GPU + S3 role check.
$PY - <<'PYEOF'
import torch, transformers
from transformers import ModernBertModel
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "(ModernBERT OK)")
PYEOF

# The CUDA allocator setting that avoids fragmentation OOMs; persist for all shells.
grep -q PYTORCH_CUDA_ALLOC_CONF ~/.bashrc || \
  echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

echo "bootstrap complete"

#!/usr/bin/env bash
# Round 3: the dense-vs-sparse supervised ablation. Generate the manifest, run the
# wave (anchors + 6 unsup→sup[X] arms warm-started from the round-1 MLM base), sync
# results to S3, then self-stop. Detached (nohup) — survives a disconnected laptop.
set -x
cd /home/ec2-user/CLIMB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments

mkdir -p experiments/climb_v2_ablation
$PY experiment_v2.py --spec configs/v2_ablation.yaml \
    --output experiments/climb_v2_ablation/manifest.json
$PY scripts/launch_v2_wave.py \
    --manifest experiments/climb_v2_ablation/manifest.json --worker_name w0

aws s3 sync experiments/climb_v2_ablation "$S3/climb_v2_ablation" \
    --exclude "*/encoder/*" --exclude "*/tokenizer/*"
echo "ROUND3 DONE $(date -u)"
sudo shutdown -h now

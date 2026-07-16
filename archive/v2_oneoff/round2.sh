#!/usr/bin/env bash
# Round 2 on an EC2 worker: Exp D (label-efficiency, reuses exploratory encoders)
# then the 3-seed headline replication, then S3 sync + self-stop. Designed to run
# detached (nohup) and survive a disconnected laptop.
set -x
cd /home/ec2-user/CLIMB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1          # ModernBERT eval must not trigger torch.compile
PY=/home/ec2-user/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments

# 1) Exp D — label-efficiency curves (no new pretraining)
$PY scripts/run_label_efficiency.py \
    --exploratory_root experiments/climb_v2 \
    --output_dir experiments/climb_v2_labeleff
aws s3 sync experiments/climb_v2_labeleff "$S3/climb_v2_labeleff" \
    --exclude "*/encoder/*" --exclude "*/tokenizer/*"

# 2) Headline — 3 pretraining seeds x {unsup,sup,mixed} + anchors
mkdir -p experiments/climb_v2_headline
$PY experiment_v2.py --spec configs/v2_headline.yaml \
    --output experiments/climb_v2_headline/manifest.json
$PY scripts/launch_v2_wave.py \
    --manifest experiments/climb_v2_headline/manifest.json --worker_name w0

# 3) Final belt-and-suspenders sync, then stop the instance
aws s3 sync experiments/climb_v2_headline "$S3/climb_v2_headline" \
    --exclude "*/encoder/*" --exclude "*/tokenizer/*"
aws s3 sync experiments/climb_v2_labeleff "$S3/climb_v2_labeleff" \
    --exclude "*/encoder/*" --exclude "*/tokenizer/*"
echo "ROUND2 DONE $(date -u)"
sudo shutdown -h now

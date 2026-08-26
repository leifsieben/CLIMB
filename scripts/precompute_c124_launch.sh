#!/usr/bin/env bash
# Launch one precompute box for a shard range. Tries every subnet, and falls back through a list of
# instance types, because capacity errors are per-AZ AND per-type.
#
# Usage: precompute_c124_launch.sh <lo> <hi> <spot|ondemand> <type> [type ...]
set -u
LO=$1; HI=$2; MARKET=$3; shift 3
AMI=ami-0578780bc4c87a97a
KEY=climb-gpu-key
SG=sg-0d11ba7811485655f
PROF=climb-ec2-s3-profile
SUBNETS="subnet-0e07b7ae383dcb680 subnet-011f8d4b0a6f00ab7 subnet-0b0a9a945de9f8648 subnet-0ee6327e8f5b315df"

UD="#!/bin/bash
su - ec2-user -c 'cd /home/ec2-user/CLIMB && git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux && setsid nohup bash scripts/precompute_c124_box.sh $LO $HI > /home/ec2-user/precompute_boot.log 2>&1 &'"

market_args=""
[ "$MARKET" = spot ] && market_args="--instance-market-options MarketType=spot"

for t in "$@"; do
  for sn in $SUBNETS; do
    id=$(aws ec2 run-instances --image-id $AMI --instance-type "$t" --key-name $KEY \
      --security-group-ids $SG --subnet-id "$sn" --iam-instance-profile Name=$PROF \
      --instance-initiated-shutdown-behavior terminate $market_args \
      --user-data "$UD" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=climb-precompute-c124-${LO}-${HI}},{Key=Shards,Value=${LO}-${HI}}]" \
      --query 'Instances[].InstanceId' --output text 2>/dev/null)
    case "$id" in i-*) echo "LAUNCHED shards $LO-$HI -> $id ($t, $MARKET, $sn)"; exit 0 ;; esac
  done
  echo "no capacity for $t ($MARKET) in any subnet" >&2
done
echo "FAILED to launch shards $LO-$HI" >&2
exit 1

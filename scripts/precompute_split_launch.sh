#!/usr/bin/env bash
# Launch ONE on-demand box to row-split a short list of blocking shards.
#
# On-demand, not spot, and that is the whole point: five spot reclaims in a day have twice
# orphaned the exact shards a rung was waiting on. A box on the critical path that cannot be
# taken away is worth more than the discount on a box that can.
set -u
LIST=$1; shift
AMI=ami-0578780bc4c87a97a
KEY=climb-gpu-key
SG=sg-0d11ba7811485655f
PROF=climb-ec2-s3-profile
SUBNETS="subnet-0e07b7ae383dcb680 subnet-011f8d4b0a6f00ab7 subnet-0b0a9a945de9f8648 subnet-0ee6327e8f5b315df"
# Commas terminate a value in the --tag-specifications shorthand, so a comma-separated shard list
# turns into a PARAMETER error -- which the old `2>/dev/null` then reported as "no capacity" for
# every instance type in the list. A launcher that misattributes its own failure sends you hunting
# for capacity in a different region while nothing is wrong with capacity at all.
TAGLIST=$(echo "$LIST" | tr ',' '-')

UD="#!/bin/bash
su - ec2-user -c 'cd /home/ec2-user/CLIMB && git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux && setsid nohup bash scripts/precompute_split_box.sh $LIST > /home/ec2-user/split_boot.log 2>&1 &'"

ERR=$(mktemp)
for t in "$@"; do
  for sn in $SUBNETS; do
    id=$(aws ec2 run-instances --image-id $AMI --instance-type "$t" --key-name $KEY \
      --security-group-ids $SG --subnet-id "$sn" --iam-instance-profile Name=$PROF \
      --instance-initiated-shutdown-behavior terminate \
      --user-data "$UD" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=climb-split-blocking},{Key=Shards,Value=${TAGLIST}}]" \
      --query 'Instances[].InstanceId' --output text 2>"$ERR")
    case "$id" in i-*) echo "LAUNCHED split box $id ($t, ondemand, $sn) for $LIST"; exit 0 ;; esac
    grep -q "InsufficientInstanceCapacity\|Unsupported" "$ERR" || { echo "NOT a capacity error:" >&2; cat "$ERR" >&2; exit 2; }
  done
  echo "no capacity for $t in any subnet" >&2
done
echo "FAILED to launch split box" >&2
exit 1

#!/usr/bin/env bash
# Launch fig_B rungs as GPU capacity frees, in Leif's priority order.
#
# ORDER IS DELIBERATE AND IS NOT WALL-CLOCK OPTIMAL. Starting the 36.5 h rung first would finish
# everything sooner. The bridge goes first because if anything dies at hour 30, the bridge plus 50M
# still answer the objective question, whereas losing the bridge makes the other two unreadable --
# skip_dense_* moving to pubchem_124m_full changes unique molecules AND SMILES notation (0.3% vs
# 86.4% lowercase-aromatic), and only the 8M bridge separates those.
#
# Capacity is checked by ATTEMPTING A LAUNCH, not by reading a quota. `run-instances --dry-run`
# reports "would have succeeded" for a request the real call rejects with VcpuLimitExceeded: DryRun
# validates permissions and parameters, never quota. It is an authorization test, not a capacity
# test.
set -u
cd "$(dirname "$0")/.."
AMI=ami-0578780bc4c87a97a
KEY=climb-gpu-key
SG=sg-0d11ba7811485655f
PROF=climb-ec2-s3-profile
SUBNETS="subnet-0e07b7ae383dcb680 subnet-011f8d4b0a6f00ab7 subnet-0b0a9a945de9f8648 subnet-0ee6327e8f5b315df"
LOG=analysis/figB_queue.log
say () { echo "[queue] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

# run_id : instance type
QUEUE="skip_dense_8M_c124:g5.4xlarge
u2s_dense_from50M:g5.4xlarge
u2s_dense_from100M:g5.4xlarge
skip_dense_50M_c124:g5.4xlarge
skip_dense_100M_c124:g5.4xlarge"

launched () {  # already running or already complete on S3?
  local r=$1
  aws ec2 describe-instances --filters "Name=tag:Run,Values=$r" \
      "Name=instance-state-name,Values=pending,running" \
      --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | grep -q i- && return 0
  aws s3 ls "s3://climb-s3-bucket/experiments/climb_v2_phase2/$r/encoder/model.safetensors" >/dev/null 2>&1
}

try_launch () {  # -> 0 if an instance actually started
  local r=$1 t=$2 sn id
  for sn in $SUBNETS; do
    id=$(aws ec2 run-instances --image-id $AMI --instance-type "$t" --key-name $KEY \
      --security-group-ids $SG --subnet-id "$sn" --iam-instance-profile Name=$PROF \
      --instance-initiated-shutdown-behavior terminate \
      --user-data "#!/bin/bash
su - ec2-user -c 'cd /home/ec2-user/CLIMB && git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux && setsid nohup bash scripts/figB_run.sh $r > /home/ec2-user/figB_boot.log 2>&1 &'" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=climb-figB-$r},{Key=Run,Value=$r}]" \
      --query 'Instances[].InstanceId' --output text 2>>"$LOG")
    case "$id" in i-*) say "LAUNCHED $r -> $id ($t, $sn)"; return 0 ;; esac
  done
  return 1
}

say "queue start"
while :; do
  pending=0
  while IFS=: read -r r t; do
    [ -n "${r:-}" ] || continue
    if launched "$r"; then continue; fi
    pending=$((pending+1))
    # STRICT ORDER: never start a later rung while an earlier one is still unlaunched.
    if try_launch "$r" "$t"; then sleep 60; else say "no capacity for $r -- holding"; fi
    break
  done <<< "$QUEUE"
  [ "$pending" -eq 0 ] && { say "QUEUE DRAINED -- every rung launched or complete"; break; }
  sleep 300
done

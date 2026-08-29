#!/usr/bin/env bash
# Launch one eval box for a space-separated list of run ids.
# Usage: figB_eval_launch.sh "<run_id> [run_id ...]" <label> [script]
#
# `script` defaults to the fig_B battery. Pass scripts/figA_rank_arm.sh to run the RANKING suites
# instead (full Polaris, CBS, Wong, FartDB) -- fig_A needs four suites the fig_B battery does not
# produce, and it rescales an arm that is short of one rather than dropping it, so the gap moves a
# rank silently instead of erroring.
set -u
RUNS=$1; LABEL=$2; SCRIPT=${3:-scripts/figB_eval_run.sh}
AMI=ami-0578780bc4c87a97a
KEY=climb-gpu-key
SG=sg-0d11ba7811485655f
PROF=climb-ec2-s3-profile
# 1a included: it was missing from every launcher here and capacity errors are per-AZ.
SUBNETS="subnet-0697512b6a144ff98 subnet-0e07b7ae383dcb680 subnet-011f8d4b0a6f00ab7 subnet-0b0a9a945de9f8648 subnet-0ee6327e8f5b315df"
# STOP, not terminate, on self-shutdown. The battery only shuts down once every artifact is
# verified present, so a shutdown is normally success -- but on 2026-08-28 an eval box shut down for
# a reason never established, and `stop` is what preserved its disk and made that a restart instead
# of a lost rung. A stopped box costs a few dollars a day and is cleaned up explicitly once the
# artifacts are checked; a terminated one takes its logs with it.
ERR=$(mktemp)

# An untagged box is invisible to every `--filters tag:Name` sweep this project uses to reconstruct
# the fleet, so it bills forever and nothing pages anyone. On 2026-08-29 a broken edit put a comment
# INSIDE a line continuation, which truncated the run-instances call before its --tag-specifications
# and --query: AWS launched a perfectly good g5.4xlarge with no tags and no user-data, the captured
# output was JSON rather than an i-..., the script reported "NOT capacity" and exited, and the box
# sat idle at ~$39/day until another session noticed it. The lesson is not "write better edits" --
# it is that the launcher must TEST for the condition rather than assume its own call was well
# formed. Sweep on every exit path.
orphan_sweep () {
  local orphans
  orphans=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=pending,running" \
              "Name=iam-instance-profile.arn,Values=*${PROF}" \
    --query 'Reservations[].Instances[?!not_null(Tags[?Key==`Name`].Value)].InstanceId' \
    --output text 2>/dev/null)
  [ -n "$orphans" ] && echo "WARNING: UNTAGGED instance(s) on $PROF -- probably a partial launch, terminate or tag them: $orphans" >&2
  return 0
}
trap orphan_sweep EXIT
for t in g5.4xlarge g5.2xlarge g6.4xlarge; do
  for sn in $SUBNETS; do
    id=$(aws ec2 run-instances --image-id $AMI --instance-type "$t" --key-name $KEY \
      --security-group-ids $SG --subnet-id "$sn" --iam-instance-profile Name=$PROF \
      --instance-initiated-shutdown-behavior stop \
      --user-data "#!/bin/bash
su - ec2-user -c 'cd /home/ec2-user/CLIMB && git fetch -q origin v2-redux && git reset -q --hard origin/v2-redux && EVAL_SHUTDOWN=1 setsid nohup bash $SCRIPT $RUNS > /home/ec2-user/eval_boot.log 2>&1 &'" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=climb-figB-eval-${LABEL}}]" \
      --query 'Instances[].InstanceId' --output text 2>"$ERR")
    case "$id" in i-*) echo "LAUNCHED eval-$LABEL -> $id ($t, $sn) for: $RUNS"; exit 0 ;; esac
    grep -q "InsufficientInstanceCapacity\|Unsupported\|VcpuLimitExceeded" "$ERR" || { echo "NOT capacity:" >&2; cat "$ERR" >&2; exit 2; }
  done
done
echo "FAILED to launch eval-$LABEL" >&2; exit 1

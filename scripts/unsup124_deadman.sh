#!/usr/bin/env bash
# Dead-man switch for the unsup124 runs.
#
# Every other alert in this project is published BY the box. That cannot tell anyone the box
# itself died -- a kernel panic, a hypervisor failure or an unexpected stop produces silence,
# and silence looks exactly like a healthy run to an email-based alerting scheme. There is
# currently no alarm of this kind in the account.
#
# This alarm watches the CLIMB/Heartbeat metric that unsup124_run.sh publishes every 60s and
# fires when it goes MISSING (TreatMissingData=breaching), so silence pages the operator.
#
#   Usage: unsup124_deadman.sh create <tag> | delete <tag>
set -euo pipefail
ACTION=${1:?create|delete}
TAG=${2:?tag}
SNS=arn:aws:sns:us-east-1:075120018132:climb-experiments
NAME="climb-heartbeat-$TAG"

if [ "$ACTION" = "delete" ]; then
  aws cloudwatch delete-alarms --region us-east-1 --alarm-names "$NAME"
  echo "deleted $NAME"
  exit 0
fi

aws cloudwatch put-metric-alarm \
  --region us-east-1 \
  --alarm-name "$NAME" \
  --alarm-description "DEAD-MAN SWITCH: $TAG stopped publishing CLIMB/Heartbeat for 30 min. The box is gone, hung, or was stopped -- training is NOT progressing. Alerts published from the box cannot report this." \
  --namespace CLIMB --metric-name Heartbeat \
  --dimensions Name=Run,Value="$TAG" \
  --statistic Sum --period 300 --evaluation-periods 6 \
  --threshold 1 --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$SNS" --ok-actions "$SNS"
echo "created $NAME (30 min of silence -> SNS page)"

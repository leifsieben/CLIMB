#!/usr/bin/env bash
# Direct box -> user notification via AWS SNS. NO Claude Code in the path: the EC2
# box itself publishes to an SNS topic whose email subscriber is the user, so run
# START / DONE / TRUNCATION / STALL / heartbeat all reach the user within minutes
# even if nothing is monitoring from outside.
#
# Usage: notify.sh <LEVEL> <subject> <message>
#   LEVEL ∈ INFO | DONE | ALERT   (ALERT = something needs attention)
#
# Never fatal: a notify failure must not take down a training run, so every path
# swallows errors and returns 0.
set -u
ARN="${CLIMB_SNS_ARN:-arn:aws:sns:us-east-1:075120018132:climb-experiments}"
REGION="${CLIMB_SNS_REGION:-us-east-1}"
LEVEL="${1:-INFO}"
SUBJECT="${2:-CLIMB}"
MSG="${3:-}"

# instance id via IMDSv2 (falls back to IMDSv1, then hostname)
TOKEN=$(curl -s --max-time 2 -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 120" 2>/dev/null)
IID=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: ${TOKEN}" \
      http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
[ -z "${IID:-}" ] && IID=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
[ -z "${IID:-}" ] && IID=$(hostname 2>/dev/null || echo "unknown-host")
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

FULL="[${LEVEL}] ${STAMP}
box: ${IID}
worker: ${CLIMB_WORKER:-?}
${MSG}"

aws sns publish --region "$REGION" --topic-arn "$ARN" \
    --subject "CLIMB ${LEVEL}: ${SUBJECT}" --message "$FULL" >/dev/null 2>&1 || true
# also drop a local breadcrumb so the box has its own audit trail
echo "$STAMP [$LEVEL] $SUBJECT :: ${MSG}" >> /home/ec2-user/CLIMB/notify.log 2>/dev/null || true
exit 0

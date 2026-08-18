#!/usr/bin/env bash
# Unattended driver for the unsup124 runs. Nobody is watching for two weeks.
#
# Deliberately does NOT use phase2_worker.sh: that script calls `sudo shutdown -h now` itself
# the moment its completion gate passes, which races (and kills) any post-training stage. The
# 5-fold CV eval is a required deliverable here, so the shutdown has to happen after it, under
# one owner. This script is that owner.
#
#   Usage: unsup124_run.sh <manifest.json> <tag> <max_hours>
#
# Exit policy, per the operator's instruction:
#   success -> sync, verify artifacts READ BACK from S3, drop the dead-man alarm, then stop
#   failure -> alert and LEAVE THE BOX UP for inspection (the climb-idle-autostop-* CloudWatch
#              alarm is the cost backstop: CPU<3% for 4h stops it anyway)
# Stop, never terminate: an unsynced disk on a terminated box is unrecoverable and unauditable.
set -uo pipefail
cd /home/ec2-user/CLIMB

MANIFEST=${1:?manifest required}
TAG=${2:?tag required}
MAX_HOURS=${3:?max_hours required}

PY=/home/ec2-user/venvs/climb/bin/python
LOG=/home/ec2-user/unsup124_${TAG}.log
export CLIMB_SNS_ARN=arn:aws:sns:us-east-1:075120018132:climb-experiments
export CLIMB_WORKER=$TAG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
NOTIFY="bash scripts/notify.sh"
TASKS="ESOL Lipophilicity QM7 BBBP BACE Tox21 HIV"

S3ROOT=$($PY -c "import json;print(json.load(open('$MANIFEST'))['s3_backup_root'])")
LOCALROOT=$($PY -c "import json;print(json.load(open('$MANIFEST'))['results_root'])")
RUNIDS=$($PY -c "import json;print(' '.join(r['run_id'] for r in json.load(open('$MANIFEST'))['runs']))")
IID=$(curl -s --max-time 3 -H "X-aws-ec2-metadata-token: $(curl -s --max-time 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 120')" http://169.254.169.254/latest/meta-data/instance-id)

say(){ echo "[$TAG $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

push_all(){
  for r in $RUNIDS; do
    [ -d "$LOCALROOT/$r" ] && aws s3 sync "$LOCALROOT/$r" "$S3ROOT/$r" \
        --exclude "*/tokenizer/*" --only-show-errors
  done
}

say "START manifest=$MANIFEST runs=[$RUNIDS] deadline=${MAX_HOURS}h box=$IID"

# ---------------- preflight gate: never launch on a bad box ----------------
if ! $PY scripts/unsup124_preflight.py "$MANIFEST" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED - not launching"
  $NOTIFY ALERT "[$TAG] preflight FAILED - nothing launched" \
    "Box $IID left UP for inspection. Log: $LOG"
  exit 1
fi

$NOTIFY INFO "[$TAG] training started" \
  "Runs: $RUNIDS. Deadline ${MAX_HOURS}h. Corpus: pubchem_124m_full_tokenized_pkl."

# ---------------- sidecar 1: dead-man heartbeat -> CloudWatch ----------------
# Alerts published FROM the box cannot tell anyone the box itself died. This metric is what a
# CloudWatch alarm with TreatMissingData=breaching watches, so SILENCE pages the operator.
( while true; do
    fp=0
    for r in $RUNIDS; do
      f="$LOCALROOT/$r/metrics.jsonl"
      [ -f "$f" ] && fp=$(( fp + $(tail -1 "$f" | $PY -c "import sys,json;print(int(json.loads(sys.stdin.read() or '{}').get('forward_passes_seen',0)))" 2>/dev/null || echo 0) ))
    done
    aws cloudwatch put-metric-data --region us-east-1 --namespace CLIMB \
        --metric-name Heartbeat --dimensions Run=$TAG --value 1 >/dev/null 2>&1
    aws cloudwatch put-metric-data --region us-east-1 --namespace CLIMB \
        --metric-name ForwardPasses --dimensions Run=$TAG --value "$fp" >/dev/null 2>&1
    sleep 60
  done ) &
HB=$!

# Arm the dead-man alarm only once the heartbeat is actually flowing, otherwise
# TreatMissingData=breaching pages the operator instantly for a run that is fine.
sleep 180
bash scripts/unsup124_deadman.sh create "$TAG" >> "$LOG" 2>&1 && \
  say "dead-man alarm climb-heartbeat-$TAG armed (30 min silence -> SNS page)" || \
  say "WARNING: could not arm dead-man alarm"

# ---------------- sidecar 2: push to S3 DURING the run, not just at the end ----------------
( while true; do sleep 900; push_all; done ) &
PUSH=$!

# ---------------- sidecar 3: hard deadline ----------------
( sleep $(( MAX_HOURS * 3600 ))
  say "DEADLINE ${MAX_HOURS}h reached - killing training and saving"
  pkill -f "launch_v2_wave.py --manifest $MANIFEST"
  pkill -f pretrain_v2.py
) &
DEADLINE=$!

cleanup_sidecars(){ kill $HB $PUSH $DEADLINE 2>/dev/null; }

# ---------------- stage 1: train + hold-out eval ----------------
# launch_v2_wave.py spawns its own per-run throughput watchdog (stall -> kill, slow -> leave
# alone) and writes verified.json ONLY when achieved forward passes >= 98% of budget.
say "stage 1: launch_v2_wave (train + moleculenet hold-out eval)"
$PY scripts/launch_v2_wave.py --manifest "$MANIFEST" --worker_name "$TAG" 2>&1 | tee -a "$LOG"
RC=$?
say "stage 1 exited rc=$RC"
push_all

# ---------------- stage 2: 5-fold CV eval (same partition as the existing ladder) ----------------
TOK=experiments/_tokenizer_unsup124
aws s3 sync s3://climb-s3-bucket/tokenizer_10M "$TOK" --only-show-errors
for r in $RUNIDS; do
  d="$LOCALROOT/$r"
  if [ ! -f "$d/verified.json" ]; then
    say "stage 2: SKIP $r (no verified.json - training did not complete)"; continue
  fi
  if [ -f "$d/moleculenet_cv/suite_summary.json" ] && \
     grep -q "HIV_nef1_MEAN" "$d/moleculenet_cv/suite_summary.json" 2>/dev/null; then
    say "stage 2: $r CV already complete"; continue
  fi
  say "stage 2: 5-fold CV eval for $r"
  rm -rf /tmp/*-featurized 2>/dev/null
  $PY eval_v2.py --output_dir "$d/moleculenet_cv" --encoder "$d/encoder" --tokenizer "$TOK" \
      --pool mean --standardize zscore --head mlp --max_length 256 \
      --head_seeds 0 1 2 --cv_folds 5 --subsample_seed 0 --datasets $TASKS 2>&1 | tee -a "$LOG"
  say "stage 2: $r CV rc=$?"
  aws s3 sync "$d/moleculenet_cv" "$S3ROOT/$r/moleculenet_cv" --only-show-errors
done

# ---------------- final sync, then PROVE it ----------------
push_all
aws s3 cp "$LOG" "$S3ROOT/_logs/$(basename "$LOG")" --only-show-errors
aws s3 cp "$MANIFEST" "$S3ROOT/_logs/$(basename "$MANIFEST")" --only-show-errors

say "verifying completion from ACHIEVED forward passes and reading artifacts back FROM S3"
$PY scripts/unsup124_verify.py "$MANIFEST" 2>&1 | tee -a "$LOG"
VRC=$?
aws s3 cp "$LOG" "$S3ROOT/_logs/$(basename "$LOG")" --only-show-errors

cleanup_sidecars

if [ "$VRC" -eq 0 ]; then
  $NOTIFY DONE "[$TAG] ALL RUNS VERIFIED COMPLETE - box stopping" \
    "Runs: $RUNIDS. Verified from achieved forward passes (>=98% of budget) AND artifacts read back from S3. Results: $S3ROOT. Box $IID stopping (NOT terminated)."
  # drop the dead-man alarm so a clean stop does not page as a lost heartbeat
  aws cloudwatch delete-alarms --region us-east-1 --alarm-names "climb-heartbeat-$TAG" 2>/dev/null
  say "stopping instance"
  sudo shutdown -h now
else
  $NOTIFY ALERT "[$TAG] INCOMPLETE - box LEFT UP for inspection" \
    "Verification failed (rc=$VRC). Everything synced to $S3ROOT and $S3ROOT/_logs/ already. Box $IID is STILL RUNNING; the climb-idle-autostop alarm will stop it after 4h of idle. Resume: aws ec2 start-instances --instance-ids $IID"
  say "verification FAILED - leaving box up"
  exit 1
fi

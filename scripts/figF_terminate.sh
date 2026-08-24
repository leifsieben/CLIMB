#!/usr/bin/env bash
# Self-terminate this box -- but ONLY after proving every artifact it holds is durable elsewhere.
#
# The instance's initiated-shutdown-behaviour was set to `terminate` from the laptop, so this needs
# no IAM widening: `shutdown -h` is enough, and the role stays S3-only.
#
# NEVER BLIND-SHUTDOWN. A box that terminates on a timer, or on idle CPU, destroys the logs and the
# partial outputs exactly when they are the only evidence of what went wrong. Every gate below is a
# TEST of a condition, not a restatement of one: object COUNTS and byte SIZES fetched from S3, not
# "the upload step ran". If any check fails this exits WITHOUT shutting down and says why.
set -u
cd /home/ec2-user/CLIMB
FIN_PID=$1
LOG=analysis/figF_terminate.log
S3=s3://climb-s3-bucket/experiments/_figF
say () { echo "[terminate] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP for inspection"; exit 1; }

say "waiting on finisher pid $FIN_PID"
while kill -0 "$FIN_PID" 2>/dev/null; do sleep 60; done
say "finisher gone"
grep -q "FINISH:" analysis/figF_passB_finish.log 2>/dev/null \
  || abort "finisher exited without writing FINISH -- uploads unproven"

# --- push the run's own logs before anything is destroyed -------------------------------------
for l in analysis/figF_stage2.log analysis/figF_passB_finish.log analysis/figF_worker2.log; do
  [ -s "$l" ] && aws s3 cp "$l" "$S3/logs/$(basename "$l")" --only-show-errors
done
aws s3 cp "$LOG" "$S3/logs/figF_terminate.log" --only-show-errors
say "logs pushed"

# --- gate 1: every EXTRA table on disk is on S3 at the same SIZE ------------------------------
n_extra=0
for f in analysis/rigor/*_EXTRA.csv; do
  [ -s "$f" ] || continue
  b=$(basename "$f"); n_extra=$((n_extra+1))
  loc=$(stat -c %s "$f")
  rem=$(aws s3api head-object --bucket climb-s3-bucket --key "experiments/_figF/$b" \
          --query ContentLength --output text 2>/dev/null)
  [ "$rem" = "$loc" ] || abort "$b is $loc bytes locally, $rem on S3"
done
[ "$n_extra" -ge 1 ] || abort "no *_EXTRA.csv on disk at all -- Pass B produced nothing"
say "gate 1 OK -- $n_extra EXTRA tables byte-matched on S3"

# --- gate 2: the twelve Pass A tables are STILL on S3 (count, not listing silence) -------------
n12=$(aws s3 ls "$S3/" | grep -cE "concat_(panels_)?(rdkit_sameenv|mordred)_(CLMunsup|CLMsup|CheMel)\.csv$")
[ "$n12" -eq 12 ] || abort "expected 12 Pass A tables on S3, counted $n12"
say "gate 2 OK -- 12 Pass A tables present"

# --- gate 3: the six Ames prediction files are on S3 ------------------------------------------
n6=$(aws s3 ls "$S3/ames/" --recursive | grep -c "test_predictions.csv$")
[ "$n6" -eq 6 ] || abort "expected 6 Ames prediction files on S3, counted $n6"
say "gate 3 OK -- 6 Ames prediction files present"

# --- gate 4: nothing else of value is single-copy here ----------------------------------------
# amesonly_*.csv are header-only BY DESIGN (Polaris withholds test labels; the predictions above
# ARE the deliverable), so they are named as known-benign rather than silently ignored.
orphans=""
for f in analysis/rigor/*.csv; do
  b=$(basename "$f")
  case "$b" in amesonly_*|BLOCKTEST.csv) continue ;; esac
  aws s3 ls "$S3/$b" >/dev/null 2>&1 || orphans="$orphans $b"
done
[ -z "$orphans" ] || abort "single-copy csv on this box:$orphans"
say "gate 4 OK -- no single-copy results"

say "ALL GATES PASSED -- terminating (instance-initiated-shutdown-behavior=terminate)"
aws s3 cp "$LOG" "$S3/logs/figF_terminate.log" --only-show-errors
sudo shutdown -h now

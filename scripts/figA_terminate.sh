#!/usr/bin/env bash
# Self-terminate the fig_A box -- but only after proving every artifact is durable on S3.
#
# instance-initiated-shutdown-behavior was set to `terminate` AT LAUNCH, so this needs no IAM
# widening: the role stays S3-only and `shutdown -h` is enough.
#
# NEVER BLIND-SHUTDOWN. Every gate below RECONCILES THE BOX AGAINST S3 rather than checking local
# completion. That distinction is not theoretical: tonight the box-side audit reported 18/18
# replicate dirs present and current while S3 held 17, because the only code path that uploads is
# the one that RUNS -- a correctly-skipped cell is never compared against the remote. The same
# shape cost 14 Ames cells when the previous box terminated.
set -u
cd /home/ec2-user/CLIMB
PY=~/venvs/climb/bin/python
S3=s3://climb-s3-bucket/experiments/figA_clms
LOG=analysis/figA_terminate.log
say () { echo "[fin] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
abort () { say "ABORT -- $* -- BOX STAYS UP"; aws s3 cp "$LOG" "$S3/logs/figA_terminate.log" --only-show-errors; exit 1; }

# EXCLUDE OUR OWN ANCESTORS. This script is launched as `bash -c "bash figA_wave3.sh; bash
# figA_terminate.sh"`, so the PARENT'S COMMAND LINE CONTAINS figA_wave3.sh -- a bare
# `pgrep -f figA_wave` matches that still-live parent and the wait never ends. The box would run
# forever having done all its work. Same self-match that made an earlier chained waiter hang, one
# level up: it is not the waiter's own cmdline this time, it is its parent's.
ancestors=""
_p=$$
while [ "$_p" -gt 1 ]; do
  ancestors="$ancestors $_p"
  _p=$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ')
  [ -n "$_p" ] || break
done
say "waiting for the work chain to drain (ignoring own ancestors:$ancestors)"
while :; do
  live=$(pgrep -f "figA_wave|wong_run.py|fartdb_multiclass.py|chemeleon_suite_run.py|eval_v2.py" 2>/dev/null)
  rest=""
  for q in $live; do
    case " $ancestors " in *" $q "*) continue ;; esac
    rest="$rest $q"
  done
  [ -z "$rest" ] && break
  sleep 60
done
say "no work processes remain"

# --- push logs first: they die with the instance and are the only evidence if a gate trips ------
for l in analysis/figA_*.log; do
  [ -s "$l" ] && aws s3 cp "$l" "$S3/logs/$(basename "$l")" --only-show-errors
done

# --- gate 1: RECONCILE every local result dir against S3 ----------------------------------------
missing=""
for tree in "chemeleon_suite/moleculeace:$S3/moleculeace" "chemeleon_suite/polaris:$S3/polaris" \
            "wong_saureus:$S3/wong" "fartdb:$S3/fartdb"; do
  loc=figure_data/${tree%%:*}; rem=${tree#*:}
  [ -d "$loc" ] || continue
  for d in "$loc"/*/; do
    [ -s "${d}verified.json" ] || continue
    n=$(basename "$d")
    if ! aws s3 ls "$rem/$n/verified.json" >/dev/null 2>&1; then
      say "UPLOADING $n (present locally, absent on S3)"
      aws s3 cp "$d" "$rem/$n" --recursive --only-show-errors || missing="$missing $n"
    fi
  done
done
[ -z "$missing" ] || abort "failed to upload:$missing"
say "gate 1 OK -- every completed local dir is on S3"

# --- gate 2: the three literature arms have THREE replicate dirs on each suite track -------------
for t in moleculeace polaris; do
  n=$(aws s3 ls "$S3/$t/" | awk '{print $2}' \
      | grep -cE "^(chemberta_mtr|molformer_c3|selfies_ted)(_s1|_s2)?/$")
  [ "$n" -eq 9 ] || abort "$t has $n of 9 literature-arm dirs on S3"
done
say "gate 2 OK -- 9 of 9 literature dirs on each suite track"

# --- gate 3: nothing of value is single-copy on this box -----------------------------------------
orph=""
for f in figure_data/_chemberta_mtr.npz figure_data/_molformer_c3.npz figure_data/_selfies_ted.npz \
         figure_data/_figA_smiles.json; do
  [ -s "$f" ] || continue
  aws s3 ls "$S3/tables/$(basename "$f")" >/dev/null 2>&1 || {
    say "uploading feature table $(basename "$f")"
    aws s3 cp "$f" "$S3/tables/$(basename "$f")" --only-show-errors || orph="$orph $f"; }
done
[ -z "$orph" ] || abort "feature tables not durable:$orph"
say "gate 3 OK -- feature tables and molecule universe are on S3"

say "ALL GATES PASSED -- terminating"
aws s3 cp "$LOG" "$S3/logs/figA_terminate.log" --only-show-errors
sudo shutdown -h now

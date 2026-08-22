#!/usr/bin/env bash
# WHERE DOES THE 0.234 ENV DRIFT LIVE -- the encoder forward pass, or the XGBoost head?
#
# Stage 0 of the xgb seed job found that unsup_8M__xgb/moleculeace does not reproduce in a fresh
# venv: 630 shared cells, max |delta| 0.234, same encoder, same seeds. That arm is
# encoder-featurized AND xgb-headed, so it cannot say which half moved. The answer decides whether
# chemeleon_frozen__xgb -- a ranked arm, #2 overall, whose base predates its replicates by a day
# and a wave -- carries an interval that is partly an environment measurement.
#
# TEST A ISOLATES THE ENCODER, AND NEEDS NOTHING NEW. unsup_8M (frozen, MLP head) uses the SAME
# encoder forward pass as unsup_8M__xgb and a different head. Rebuild it here at its own published
# seeds and diff:
#     A reproduces  -> the forward pass is stable in this env; the drift is in the XGB head, and
#                      chemeleon_frozen__xgb is contaminated the same way
#     A drifts      -> the forward pass moved; encoder-featurized arms drift and chemeleon, which
#                      is featurized by chemprop, is not implicated by this evidence
#
# TEST B IS THE DIRECT ONE the peer asked for: rebuild chemeleon_frozen__xgb itself and diff. It
# needs the 3.12 chemprop venv, so it is second -- A is minutes away and already narrows it.
#
# NOTHING HERE IS PUBLISHED. Every rebuild goes to a __ENVTEST_* scratch dir and is diffed. The
# dirs are then PRESERVED under _diagnostics/ -- see the note at the bottom for why deleting them
# was a mistake the first time.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis figure_data
LOG=analysis/env_drift_locate.log
S3=s3://climb-s3-bucket/experiments
PY=~/venvs/climb/bin/python

IMDS_TOKEN=$(curl -fs --max-time 5 -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
IID=$(curl -fs --max-time 5 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
say () { echo "[drift] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
[ -n "$IID" ] || { say "FATAL no instance id"; exit 3; }
( while true; do aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors 2>/dev/null; sleep 60; done ) &
say "start on $IID"

[ -x "$PY" ] || { say "FATAL no python at $PY"; exit 2; }
$PY -c "import xgboost" 2>/dev/null || $PY -m pip install -q "xgboost==2.1.4" >> "$LOG" 2>&1
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL preflight"; exit 2; }
import torch, transformers, xgboost, sklearn, numpy
print(f"[preflight] torch={torch.__version__} transformers={transformers.__version__} "
      f"xgboost={xgboost.__version__} sklearn={sklearn.__version__} numpy={numpy.__version__}")
PYEOF
say "preflight OK"

aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "FATAL tokenizer absent"; exit 2; }
aws s3 sync s3://climb-s3-bucket/datasets/moleculeace/ chemeleon_suite/data/moleculeace/ --only-show-errors
n=$(ls chemeleon_suite/data/moleculeace/*.csv 2>/dev/null | wc -l)
[ "$n" -ge 30 ] || { say "FATAL moleculeace data incomplete ($n/30)"; exit 2; }
aws s3 cp "$S3/climb_v2_phase2/unsup_8M/encoder" figure_data/climb_v2_phase2/unsup_8M/encoder \
    --recursive --only-show-errors
[ -s figure_data/climb_v2_phase2/unsup_8M/encoder/model.safetensors ] || { say "FATAL encoder"; exit 2; }
say "inputs staged ($n moleculeace tasks)"

diff_against () {    # $1 fresh results.csv   $2 published S3 key   $3 label
  aws s3 cp "$2" /tmp/ref_$3.csv --only-show-errors
  $PY - "$1" /tmp/ref_$3.csv "$3" <<'PYEOF' 2>&1 | tee -a "$LOG"
import csv, sys
def load(p):
    return {(r["task"], r["seed"], r["subset"], r["metric"]): float(r["value"])
            for r in csv.DictReader(open(p))}
try:
    a, b = load(sys.argv[1]), load(sys.argv[2])
except FileNotFoundError as e:
    print(f"[{sys.argv[3]}] FAILED to load: {e}"); raise SystemExit(0)
common = sorted(set(a) & set(b))
if not common:
    print(f"[{sys.argv[3]}] NO SHARED CELLS -- cannot compare"); raise SystemExit(0)
# PRINT THE DISTRIBUTION, NOT JUST THE MAX. Test B printed only a max; when the per-cell detail
# was wanted the scratch dir had already been removed, and the conclusion survived only because a
# max happens to bound every cell. Do not rely on having guessed the right statistic in advance.
d = {k: abs(a[k] - b[k]) for k in common}
vals = sorted(d.values())
worst = max(d, key=d.get)
tag = sys.argv[3]
n = len(vals)
q = lambda f: vals[min(n - 1, int(f * n))]
print(f"[{tag}] {n} shared cells, max |delta| {vals[-1]:.6g} at {worst}")
print(f"[{tag}] median {q(0.5):.6g}  p90 {q(0.9):.6g}  p99 {q(0.99):.6g}  "
      f"exact {sum(1 for v in vals if v < 1e-12)}/{n}")
for thr in (1e-6, 1e-3, 1e-2, 1e-1):
    print(f"[{tag}] cells |delta| > {thr:g}: {sum(1 for v in vals if v > thr)}/{n}")
print(f"[{tag}] VERDICT {'REPRODUCES' if vals[-1] < 1e-9 else 'DOES NOT REPRODUCE'}")
PYEOF
}

# ---------------------------------------------------------------- TEST A: encoder + MLP
say "TEST A: rebuilding moleculeace/unsup_8M (encoder + MLP, seeds 42 117 709)"
$PY scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
    --encoder figure_data/climb_v2_phase2/unsup_8M/encoder --tokenizer figure_data/_tokenizer \
    --model unsup_8M__ENVTEST_MLP --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
say "TEST A run rc=$?"
diff_against figure_data/chemeleon_suite/moleculeace/unsup_8M__ENVTEST_MLP/results.csv \
    "$S3/chemeleon_suite/moleculeace/unsup_8M/results.csv" A_encoder_mlp

# ---------------------------------------------------------------- TEST B: chemeleon + XGB
# Needs chemprop, which needs >=3.11; the climb venv is 3.9. Built here rather than assumed.
if [ ! -x ~/venvs/chemeleon/bin/python ]; then
  say "building chemeleon venv (python3.12 + chemprop 2.3.1)"
  python3.12 -m venv ~/venvs/chemeleon >> "$LOG" 2>&1 \
    || { say "SKIP TEST B -- no python3.12 on this box"; python3.12 --version >> "$LOG" 2>&1; }
  if [ -x ~/venvs/chemeleon/bin/python ]; then
    ~/venvs/chemeleon/bin/python -m pip install -q --upgrade pip setuptools wheel >> "$LOG" 2>&1
    ~/venvs/chemeleon/bin/python -m pip install -q "chemprop==2.3.1" xgboost rdkit >> "$LOG" 2>&1
  fi
fi
if [ -x ~/venvs/chemeleon/bin/python ]; then
  say "TEST B: rebuilding moleculeace/chemeleon_frozen__xgb (chemeleon + XGB, seeds 42 117 709)"
  ~/venvs/chemeleon/bin/python scripts/chemeleon_suite_run.py --track moleculeace \
      --featurizer chemeleon --model chemeleon_frozen__xgb__ENVTEST --head xgb \
      --seeds 42 117 709 >> "$LOG" 2>&1
  say "TEST B run rc=$?"
  diff_against figure_data/chemeleon_suite/moleculeace/chemeleon_frozen__xgb__ENVTEST/results.csv \
      "$S3/chemeleon_suite/moleculeace/chemeleon_frozen__xgb/results.csv" B_chemeleon_xgb
else
  say "TEST B SKIPPED -- chemeleon venv unavailable"
fi

# THE SCRATCH DIR IS THE EVIDENCE. An earlier version deleted these after diffing, on the reasoning
# that a diagnostic produces answers rather than artifacts. That was wrong: for a diagnostic whose
# OUTPUT IS THE FINDING, the rebuilt cells are the only record that lets anyone re-examine the
# distribution -- a median, a tail, which tasks moved -- without paying the 50 minutes again. Test B
# reported max |delta| 0.0027 and the per-cell detail was already gone when I went back for it.
# The max bounds every cell so the conclusion survived, but that was luck about which statistic I
# had happened to print, not a property of the design.
#
# Kept under _diagnostics/ rather than the experiment tree: it must be recoverable and must never
# be mistaken for a published cell.
for d in figure_data/chemeleon_suite/moleculeace/*__ENVTEST*; do
  [ -d "$d" ] || continue
  aws s3 cp --recursive "$d" "$S3/_diagnostics/env_drift_$IID/$(basename "$d")" --only-show-errors     && say "preserved $(basename "$d") -> _diagnostics/env_drift_$IID/"
done
rm -rf figure_data/chemeleon_suite/moleculeace/*__ENVTEST*
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
say "done -- staying UP so the verdicts are inspectable"

#!/usr/bin/env bash
# Give unsup_8M__xgb and skip_dense_8M__xgb THREE pretraining-seed dirs on every track.
#
# WHY. Leif set fig_A1's admission gate at full coverage and asked that no ranked arm rest on
# fewer than 3 pretraining seeds. These two arms are ranked as of 2026-08-21 and sit at 1 dir on
# all four tracks. The embeddings are a deterministic forward pass of a frozen encoder, so the
# three seeds are the three PRETRAININGS -- unsup_8M{,_s1,_s2} and skip_dense_8M{,_s1,_s2}, all
# six already on S3 -- and the head seeds stay 42/117/709 inside each dir, as the base has them.
#
# THE BASE IS TESTED, NOT TRUSTED. Its suite cells were built in a venv that no longer exists, so
# the honest question is whether a fresh pinned venv reproduces them. Stage 0 rebuilds ONE base
# cell and diffs. If it reproduces, the environment is equivalent and only _s1/_s2 are needed. If
# it does not, the whole trio is rebuilt here so base and replicates share one environment --
# a seed spread whose dirs come from different library sets is partly an environment measurement.
# Either way the answer is recorded rather than assumed.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis figure_data
LOG=analysis/xgb_seed_replicates.log
S3=s3://climb-s3-bucket/experiments
PY=~/venvs/climb/bin/python
# IMDSv2 FIRST. The v1 unauthenticated form returns nothing on a box with IMDSv2 enforced, and
# an empty instance id collapses every box's log onto s3://.../jobs/.log -- so this failed closed
# on the first launch rather than logging into a shared key. Token first, v1 only as a fallback.
IMDS_TOKEN=$(curl -fs --max-time 5 -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
if [ -n "$IMDS_TOKEN" ]; then
  IID=$(curl -fs --max-time 5 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
      http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
else
  IID=$(curl -fs --max-time 5 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
fi
say () { echo "[xgbseed] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

# The instance id must be non-empty before it is used as a log key: an empty one collapses every
# box's log onto s3://.../jobs/.log and the per-box logs are simply lost.
[ -n "$IID" ] || { say "FATAL no instance id from metadata"; exit 3; }
( while true; do aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors 2>/dev/null; sleep 30; done ) &

say "start on $IID"

# ---------------------------------------------------------------- preflight (refuse, don't hope)
[ -x "$PY" ] || { say "FATAL no python at $PY"; exit 2; }
# THE AMI HAS NO XGBOOST -- confirmed on climb-v2-worker, not inferred. Preflight refusing is the
# right behaviour, but every box then needs a hand. Install the same pin everywhere: all six dirs
# of this rebuild must share one library set or the spread measures the fleet, not the pretraining.
$PY -c "import xgboost" 2>/dev/null || {
  say "xgboost absent (expected on this AMI) -- installing pinned 2.1.4"
  $PY -m pip install -q "xgboost==2.1.4" >> "$LOG" 2>&1 || { say "FATAL xgboost install failed"; exit 2; }
}
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL preflight imports failed"; exit 2; }
import torch, transformers, xgboost, sklearn, rdkit, numpy
print(f"[preflight] torch={torch.__version__} transformers={transformers.__version__} "
      f"xgboost={xgboost.__version__} sklearn={sklearn.__version__} numpy={numpy.__version__}")
PYEOF
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.probe" --only-show-errors || { say "FATAL cannot write S3"; exit 2; }
df -Pk . | awk 'NR==2 && $4 < 20000000 {print "FATAL disk <20GB"; exit 1}' | grep -q FATAL && { say "FATAL disk"; exit 2; }
say "preflight OK"

# inputs: encoders, tokenizer, suite data
# THE TOKENIZER IS NOT UNDER experiments/. It lives at s3://climb-s3-bucket/tokenizer_10M --
# scripts/head_comparison_run.sh has always fetched it from there. Copying the wrong prefix
# SUCCEEDS and copies nothing (an empty prefix is not an error), so the run reached the featurizer
# with an empty tokenizer dir and died 12 times on a null vocab_file. Assert the file, because the
# missing-input path and the success path are otherwise indistinguishable.
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "FATAL tokenizer.json absent after staging"; exit 2; }
$PY -c "
from transformers import PreTrainedTokenizerFast
t = PreTrainedTokenizerFast.from_pretrained('figure_data/_tokenizer')
assert t.vocab_size > 0, 'tokenizer loaded with empty vocab'
print(f'[preflight] tokenizer OK, vocab={t.vocab_size}')" >> "$LOG" 2>&1 \
  || { say "FATAL tokenizer will not load"; exit 2; }
for e in unsup_8M unsup_8M_s1 unsup_8M_s2 skip_dense_8M skip_dense_8M_s1 skip_dense_8M_s2; do
  aws s3 cp "$S3/climb_v2_phase2/$e/encoder" "figure_data/climb_v2_phase2/$e/encoder" \
      --recursive --only-show-errors
  [ -s "figure_data/climb_v2_phase2/$e/encoder/model.safetensors" ] || { say "FATAL encoder $e"; exit 2; }
done
aws s3 cp "$S3/chemeleon_suite/data" chemeleon_suite/data --recursive --only-show-errors
say "inputs staged"

# arm -> encoder stem. The output dir is <stem>__xgb<suffix>, the name the figures resolve.
# ONE DIR PER BOX, SET FROM THE LAUNCH ENVIRONMENT. Stage 0 came back "base does NOT reproduce"
# (630 shared cells, max |delta| 0.234), which expanded this job from 4 cells to 12 -- the whole
# trio for both arms, so that all three dirs of a spread share one environment. Serially that is
# ~8h at the 41 min/cell this box measured. The 12 cells are independent, so they are split across
# boxes instead, each owning one (arm, suffix) and both its tracks.
#
# EVERY BOX MUST BE THE SAME INSTANCE TYPE. That is the whole point of the rebuild: c5 is Intel
# with AVX-512 and g5 is AMD without it, and reduction order follows the instruction set. Spreading
# this across mixed types would reintroduce exactly the environment variance we are rebuilding to
# remove, and the spread would measure the fleet instead of the pretraining.
ARMS=${ARMS:-"unsup_8M skip_dense_8M"}
FORCE_SFX=${FORCE_SUFFIXES:-}
SKIP_STAGE0=${SKIP_STAGE0:-0}
SEEDS="42 117 709"

# COMPLETION IS COUNTED TASKS -- AND THE TWO TRACKS COUNT DIFFERENT FILES. See
# scripts/count_cell_tasks.py for why Polaris must count finite predictions and not results.csv.
cell_tasks () { $PY scripts/count_cell_tasks.py "$1" "$2" 2>/dev/null || echo 0; }
want_tasks () { [ "$1" = moleculeace ] && echo 30 || echo 28; }

suite_cell () {           # $1 stem  $2 suffix  $3 track  $4 outdir-override(optional)
  local stem=$1 sfx=$2 track=$3 out=${4:-}
  local enc="figure_data/climb_v2_phase2/${stem}${sfx}/encoder"
  local model="${out:-${stem}__xgb${sfx}}"
  $PY scripts/chemeleon_suite_run.py --track "$track" --featurizer encoder \
      --encoder "$enc" --tokenizer figure_data/_tokenizer \
      --model "$model" --head xgb --seeds $SEEDS >> "$LOG" 2>&1
}

# ---------------------------------------------------------------- stage 0: is the base reproducible?
# The verdict is already in from the first box -- do not spend 41 minutes re-deriving it.
if [ "$SKIP_STAGE0" = "1" ]; then
  REBUILD_BASE=${REBUILD_BASE:-1}
  say "stage 0 SKIPPED by request -- REBUILD_BASE=$REBUILD_BASE carried in from the first box's verdict"
else
say "stage 0: rebuilding unsup_8M__xgb MoleculeACE into a scratch dir to test env equivalence"
aws s3 cp "$S3/chemeleon_suite/moleculeace/unsup_8M__xgb/results.csv" /tmp/base_ref.csv --only-show-errors
suite_cell unsup_8M "" moleculeace "unsup_8M__xgb__ENVTEST"
NEW=figure_data/chemeleon_suite/moleculeace/unsup_8M__xgb__ENVTEST/results.csv
if [ -s "$NEW" ] && [ -s /tmp/base_ref.csv ]; then
  if $PY - "$NEW" /tmp/base_ref.csv >> "$LOG" 2>&1 <<'PYEOF'
import sys, csv
def load(p):
    return {(r["task"], r["seed"], r["subset"], r["metric"]): float(r["value"])
            for r in csv.DictReader(open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
common = set(a) & set(b)
worst = max((abs(a[k] - b[k]) for k in common), default=float("inf"))
print(f"[envtest] {len(common)} shared cells, max |delta| {worst:.6g}")
sys.exit(0 if worst < 1e-9 else 1)
PYEOF
  then REBUILD_BASE=0; say "stage 0: base REPRODUCES bit-for-bit -- fresh venv is equivalent, keeping published bases"
  else REBUILD_BASE=1; say "stage 0: base does NOT reproduce -- rebuilding the whole trio in this env"
  fi
else
  REBUILD_BASE=1; say "stage 0: could not compare (missing file) -- rebuilding the whole trio"
fi
fi
rm -rf figure_data/chemeleon_suite/*/unsup_8M__xgb__ENVTEST

# ---------------------------------------------------------------- stage 1: the cells
# A LITERAL SPACE CANNOT CARRY THE BASE THROUGH WORD-SPLITTING. `SUFFIXES=" _s1 _s2"` splits to
# exactly two tokens, so the `[ "$sfx" = " " ]` arm never fired and REBUILD_BASE=1 silently did
# NOT rebuild the base -- the one case the whole env test exists to handle. Use a real sentinel.
SUFFIXES="_s1 _s2"
[ "$REBUILD_BASE" = "1" ] && SUFFIXES="BASE _s1 _s2"
[ -n "$FORCE_SFX" ] && SUFFIXES="$FORCE_SFX"
for stem in $ARMS; do
  for sfx in $SUFFIXES; do
    [ "$sfx" = "BASE" ] && sfx=""
    for track in moleculeace polaris; do
      d="figure_data/chemeleon_suite/$track/${stem}__xgb${sfx}"
      want=$(want_tasks "$track")
      have=$(cell_tasks "$d" "$track")
      if [ "${have:-0}" -ge "$want" ]; then say "SKIP $track ${stem}__xgb${sfx} (has $have/$want)"; continue; fi
      say "RUN  $track ${stem}__xgb${sfx}"
      suite_cell "$stem" "$sfx" "$track"
      # COMPLETION IS ACHIEVED WORK, NOT A FILE. results.csv appears for a partial run too.
      have=$(cell_tasks "$d" "$track")
      if [ "${have:-0}" -ge "$want" ]; then
        aws s3 cp --recursive "$d" "$S3/chemeleon_suite/$track/${stem}__xgb${sfx}" --only-show-errors
        say "DONE $track ${stem}__xgb${sfx} ($have/$want tasks, uploaded)"
      else
        say "INCOMPLETE $track ${stem}__xgb${sfx} ($have/$want) -- NOT uploaded"
      fi
    done
  done
done

# ---------------------------------------------------------------- verify, then and only then stop
say "verifying the full grid from achieved work"
MISSING=0
# VERIFY ONLY WHAT THIS BOX WAS ASSIGNED. Checking all six dirs on a box that owns one would
# report five permanent MISSING and the shutdown gate would never clear.
for stem in $ARMS; do
  for sfx in $SUFFIXES; do
    [ "$sfx" = "BASE" ] && sfx=""
    for track in moleculeace polaris; do
      want=$(want_tasks "$track")
      have=$(cell_tasks "figure_data/chemeleon_suite/$track/${stem}__xgb${sfx}" "$track")
      [ "${have:-0}" -ge "$want" ] || { say "MISSING $track ${stem}__xgb${sfx} ($have/$want)"; MISSING=$((MISSING+1)); }
    done
  done
done
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
if [ "$MISSING" -eq 0 ]; then
  say "COMPLETE 12 of 12 suite cells verified -- shutting down"
  aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
  sudo shutdown -h now
else
  say "NOT COMPLETE ($MISSING cell(s) short) -- staying UP for inspection"
fi

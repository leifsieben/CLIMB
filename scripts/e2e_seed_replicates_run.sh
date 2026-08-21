#!/usr/bin/env bash
# Give unsup_8M_e2e and skip_dense_8M_e2e THREE pretraining seeds on the two SUITE tracks.
#
# WHY. fig_A1's admission gate is full coverage, and no ranked arm may rest on fewer than 3
# pretraining seeds. These two e2e arms are ranked and sit at one dir on MoleculeACE and Polaris
# (their MolNet and CBS cells already have three). Four dirs x 58 tasks x 3 seeds.
#
# THE REPLICATE AXIS IS THE ENCODER, NOT THE HEAD SEED. This is the one design decision here and
# it is easy to get backwards. The frozen replicates that already exist -- unsup_8M_s1/_s2 and
# skip_dense_8M_s1/_s2 -- hold their head seeds at 42/117/709 in ALL THREE dirs and vary the
# PRETRAINING. That is what makes them three pretraining seeds. The CheMeleon arm used disjoint
# head-seed triples instead, but CheMeleon has exactly one pretrained model, so seeds were the
# only axis it had; copying that convention here would vary the head while leaving the
# pretraining fixed, and would not satisfy the gate at all. So:
#
#     dir unsup_8M_e2e_s1   <-  encoder unsup_8M_s1      head seeds 42/117/709
#
# ONE DIR PER BOX. Each box owns exactly one (arm, suffix) and both of its tracks, set through
# ARM/SFX in the launch environment. Four boxes run disjoint dirs, so there is no collision and
# any one can be relaunched alone.
set -u
cd /home/ec2-user/CLIMB

ARM=${ARM:?ARM must be set (unsup_8M | skip_dense_8M)}
SFX=${SFX:?SFX must be set (_s1 | _s2)}
SEEDS="42 117 709"
DIRNAME="${ARM}_e2e${SFX}"
ENC="figure_data/climb_v2_phase2/${ARM}${SFX}/encoder"

mkdir -p analysis figure_data
LOG="analysis/e2e_seed_${DIRNAME}.log"
S3=s3://climb-s3-bucket/experiments
PY=~/venvs/climb/bin/python

IMDS_TOKEN=$(curl -fs --max-time 5 -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
IID=$(curl -fs --max-time 5 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
say () { echo "[e2eseed] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
[ -n "$IID" ] || { say "FATAL no instance id from metadata"; exit 3; }
( while true; do aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors 2>/dev/null; sleep 60; done ) &

say "start on $IID -- dir $DIRNAME from encoder ${ARM}${SFX}, seeds $SEEDS"

# ---------------------------------------------------------------- preflight (refuse, don't hope)
[ -x "$PY" ] || { say "FATAL no python at $PY"; exit 2; }
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "FATAL preflight imports / no GPU"; exit 2; }
import torch, transformers, sklearn, rdkit, numpy
assert torch.cuda.is_available(), "CUDA not available -- this is a fine-tuning job, refuse to run on CPU"
print(f"[preflight] torch={torch.__version__} cuda={torch.version.cuda} "
      f"gpu={torch.cuda.get_device_name(0)} transformers={transformers.__version__}")
PYEOF
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.probe" --only-show-errors || { say "FATAL cannot write S3"; exit 2; }
df -Pk . | awk 'NR==2 && $4 < 20000000 {print "FATAL disk"; exit 1}' | grep -q FATAL && { say "FATAL disk"; exit 2; }
say "preflight OK"

# ---------------------------------------------------------------- inputs (assert every one)
# The tokenizer is at tokenizer_10M, NOT under experiments/. `aws s3 cp --recursive` on a prefix
# that does not exist copies nothing and EXITS 0 -- that is how the xgb job burned an hour with an
# empty tokenizer dir and 12 identical deaths in convert_slow_tokenizer. Assert it, and load it,
# because a present file that will not load is the next variant of the same failure.
aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors
[ -s figure_data/_tokenizer/tokenizer.json ] || { say "FATAL tokenizer.json absent"; exit 2; }
$PY -c "
from transformers import PreTrainedTokenizerFast
t = PreTrainedTokenizerFast.from_pretrained('figure_data/_tokenizer')
assert t.vocab_size > 0, 'tokenizer loaded with empty vocab'
print(f'[preflight] tokenizer OK, vocab={t.vocab_size}')" >> "$LOG" 2>&1 \
  || { say "FATAL tokenizer will not load"; exit 2; }

aws s3 cp "$S3/climb_v2_phase2/${ARM}${SFX}/encoder" "$ENC" --recursive --only-show-errors
[ -s "$ENC/model.safetensors" ] || { say "FATAL encoder ${ARM}${SFX} absent"; exit 2; }
aws s3 cp "$S3/chemeleon_suite/data" chemeleon_suite/data --recursive --only-show-errors
say "inputs staged"

# ---------------------------------------------------------------- the two tracks
# COMPLETION IS COUNTED TASKS AND THE TRACKS COUNT DIFFERENT FILES -- see count_cell_tasks.py.
# Polaris results.csv is header-only by design, so counting it there returns 0 forever: it would
# withhold a finished dir and hang this shutdown gate. Never gate on a file existing; the runner
# writes results.csv after every task now, so a partial dir has one too.
for track in moleculeace polaris; do
  d="figure_data/chemeleon_suite/$track/$DIRNAME"
  want=$($PY scripts/count_cell_tasks.py --want "$track")
  have=$($PY scripts/count_cell_tasks.py "$d" "$track")
  if [ "${have:-0}" -ge "$want" ]; then say "SKIP $track $DIRNAME (has $have/$want)"; continue; fi
  say "RUN  $track $DIRNAME ($have/$want done)"
  $PY scripts/chemeleon_suite_e2e.py --track "$track" --model "$ARM" --suffix "_e2e${SFX}" \
      --encoder "$ENC" --tokenizer figure_data/_tokenizer --seeds $SEEDS >> "$LOG" 2>&1
  have=$($PY scripts/count_cell_tasks.py "$d" "$track")
  if [ "${have:-0}" -ge "$want" ]; then
    aws s3 cp --recursive "$d" "$S3/chemeleon_suite/$track/$DIRNAME" --only-show-errors
    say "DONE $track $DIRNAME ($have/$want tasks, uploaded)"
  else
    # Push the partial anyway, under a name that cannot be mistaken for a finished dir. Hours of
    # fine-tuning are worth keeping, and a resume reads it; the gate above still says NOT DONE.
    aws s3 cp --recursive "$d" "$S3/chemeleon_suite/$track/${DIRNAME}_PARTIAL" --only-show-errors
    say "INCOMPLETE $track $DIRNAME ($have/$want) -- parked as ${DIRNAME}_PARTIAL, NOT published"
  fi
done

# ---------------------------------------------------------------- verify, then and only then stop
say "verifying from achieved work"
MISSING=0
for track in moleculeace polaris; do
  want=$($PY scripts/count_cell_tasks.py --want "$track")
  have=$($PY scripts/count_cell_tasks.py "figure_data/chemeleon_suite/$track/$DIRNAME" "$track")
  [ "${have:-0}" -ge "$want" ] || { say "MISSING $track $DIRNAME ($have/$want)"; MISSING=$((MISSING+1)); }
done
aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
if [ "$MISSING" -eq 0 ]; then
  say "COMPLETE both tracks of $DIRNAME verified -- shutting down"
  aws s3 cp "$LOG" "$S3/_logs/jobs/$IID.log" --only-show-errors
  sudo shutdown -h now
else
  say "NOT COMPLETE ($MISSING track(s) short) -- staying UP for inspection"
fi

#!/bin/bash
# Experiment A FINALIZE (runs ON the box, detached). Replaces box_selfstop.sh: after the bigram wave
# verifies complete, it (1) syncs the NEW arm encoders to S3 so the checkpoints are durable and
# HF-uploadable (the wave only backs up eval results, not weights), then (2) stops the box.
# shutdown-behavior=stop -> EBS preserved. Never stops on failure (no marker -> keeps waiting).
set -uo pipefail
cd /home/ec2-user/CLIMB
S3=s3://climb-s3-bucket/experiments/climb_v2_expA
LOG=/home/ec2-user/synth/expA_finalize.log
# the 8 NEW checkpoints (baselines are re-evals of phase2 encoders already on S3 — not re-uploaded)
ENC_RUNS="unigram_8M unigram_8M_s1 unigram_8M_s2 corrupt_mlm_8M_s1 corrupt_mlm_8M_s2 bigram_8M bigram_8M_s1 bigram_8M_s2"
say(){ echo "[finalize $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "armed: after EXPA_BIGRAM_DONE -> sync encoders to S3 -> stop box"
while ! grep -q EXPA_BIGRAM_DONE /home/ec2-user/synth/expA_bigram_run.log 2>/dev/null; do sleep 300; done
say "bigram wave complete; syncing $(echo $ENC_RUNS | wc -w) encoders to S3"

ok=0
for RID in $ENC_RUNS; do
  ED=experiments/climb_v2_expA/$RID/encoder
  if [ -f "$ED/model.safetensors" ]; then
    aws s3 cp "$ED" "$S3/$RID/encoder" --recursive --only-show-errors \
      && { say "  synced $RID/encoder"; ok=$((ok+1)); } || say "  FAILED sync $RID/encoder"
  else
    say "  MISSING $RID/encoder (run incomplete?)"
  fi
done
say "encoder sync: $ok/$(echo $ENC_RUNS | wc -w) on S3 under $S3/<run>/encoder"
bash scripts/notify.sh DONE "ExpA FINALIZED — checkpoints on S3 ($ok/8)" \
  "encoders at $S3/<run>/encoder. Box stopping in 3 min. NEXT (local session): HF upload + docs + zip via scripts/publish_expA_hf.py & scripts/package_expA_bundle.py." || true
echo "EXPA_FINALIZED encoders=$ok/8"
sleep 180
sudo shutdown -h now

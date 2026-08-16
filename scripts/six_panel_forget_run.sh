#!/usr/bin/env bash
# Catastrophic-forgetting arm: sup2unsup:dense = the already-trained supervised-first 8M encoder
# (skip_dense_8M) + a 2M-FP MLM continuation (the mirror of unsup2sup:dense). 3 seeds. Then the
# 6-panel eval. Gated self-shutdown only when all 3 runs have pretrain+MoleculeNet+MoleculeACE+CBS.
set -u
cd /home/ec2-user/CLIMB
mkdir -p analysis experiments/climb_v2_phase2 configs
LOG=analysis/forget_run.log
echo "[forget] start $(date -u +%FT%TZ)" >> "$LOG"

# stage standard tokenizer (frozen eval) + descriptor/eval configs (startup validation safety)
[ -f figure_data/_tokenizer/tokenizer.json ] || { mkdir -p figure_data/_tokenizer; aws s3 sync s3://climb-s3-bucket/tokenizer_10M figure_data/_tokenizer --only-show-errors; }
[ -f configs/descriptor_stats.json ] || aws s3 cp s3://climb-s3-bucket/configs/descriptor_stats.json configs/descriptor_stats.json --only-show-errors

# warm-start encoders: the supervised-first 8M stage (already trained), one per seed
for a in skip_dense_8M skip_dense_8M_s1 skip_dense_8M_s2; do
  d=experiments/climb_v2_phase2/$a/encoder
  [ -f "$d/model.safetensors" ] || { mkdir -p "$d"; aws s3 sync "s3://climb-s3-bucket/experiments/climb_v2_phase2/$a/encoder" "$d" --only-show-errors; }
done

# Stage B: pretrain 2M MLM x3 (+ MoleculeNet auto-eval + S3 backup) via the standard launcher
~/venvs/climb/bin/python scripts/launch_v2_wave.py \
  --manifest experiments/climb_v2_phase2/manifests/s2u_seeds.json --worker_name box_forget >> "$LOG" 2>&1
echo "[forget] pretrain wave exit rc=$? $(date -u +%FT%TZ)" >> "$LOG"

# 6-panel completion: MoleculeACE (macro-RMSE) + CBS (NEF1) frozen on each new encoder
for s in s0 s1 s2; do
  run=s2u_dense_from8M_$s
  enc=experiments/climb_v2_phase2/$run/encoder
  [ -f "$enc/model.safetensors" ] || { echo "[forget] MISSING $enc" >> "$LOG"; continue; }
  if [ ! -f "figure_data/chemeleon_suite/moleculeace/$run/verified.json" ]; then
    ~/venvs/climb/bin/python scripts/chemeleon_suite_run.py --track moleculeace --featurizer encoder \
      --model "$run" --encoder "$enc" --tokenizer figure_data/_tokenizer --head mlp --seeds 42 117 709 >> "$LOG" 2>&1
    aws s3 cp --recursive "figure_data/chemeleon_suite/moleculeace/$run" "s3://climb-s3-bucket/experiments/chemeleon_suite/moleculeace/$run" --only-show-errors
  fi
  if [ ! -f "figure_data/cbs_benchmark/$run/moleculenet_cv/suite_summary.json" ]; then
    ~/venvs/climb/bin/python eval_v2.py --encoder "$enc" --tokenizer figure_data/_tokenizer \
      --output_dir "figure_data/cbs_benchmark/$run/moleculenet_cv" --head mlp --head_seeds 0 1 2 \
      --task_csv data/cbs.csv --task_name cbs --task_type classification --cv_folds 5 --cv_scheme provided >> "$LOG" 2>&1
    aws s3 cp --recursive "figure_data/cbs_benchmark/$run" "s3://climb-s3-bucket/experiments/cbs_benchmark/$run" --only-show-errors
  fi
done

done=0
for s in s0 s1 s2; do
  run=s2u_dense_from8M_$s
  [ -f "experiments/climb_v2_phase2/$run/moleculenet/suite_summary.json" ] \
    && [ -f "figure_data/chemeleon_suite/moleculeace/$run/verified.json" ] \
    && [ -f "figure_data/cbs_benchmark/$run/moleculenet_cv/suite_summary.json" ] && done=$((done+1))
done
echo "[forget] complete runs: $done/3 $(date -u +%FT%TZ)" >> "$LOG"
if [ "$done" -eq 3 ]; then
  touch figure_data/SUP2UNSUP_DONE
  echo "[forget] all done -> shutdown" >> "$LOG"; sudo shutdown -h now
else
  echo "[forget] NOT all done -> staying UP for inspection" >> "$LOG"
fi

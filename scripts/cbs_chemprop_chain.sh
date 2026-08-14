#!/usr/bin/env bash
# Detached CBS chain: wait for the tuned-Polaris e2e job to release the GPU, then run the two
# CheMeleon/chemprop comparators on the CBS benchmark (provided folds, NEF1%). Idempotent — each
# sub-runner skips already-verified work. Launched with setsid+nohup so it survives the ssh session.
set -u
cd /home/ec2-user/CLIMB
LOG=analysis/cbs_chain.log
echo "[cbs-chain] start $(date -u +%FT%TZ)" >> $LOG

# 1. wait for the tuned-Polaris job to finish (frees the T4). Cap the wait at 2h as a safety net.
for i in $(seq 1 120); do
  if grep -q TUNED_DONE analysis/e2e_tuned.log 2>/dev/null; then echo "[cbs-chain] TUNED_DONE seen" >> $LOG; break; fi
  if ! pgrep -f "chemeleon_suite_e2e.py --track polaris" >/dev/null 2>&1; then echo "[cbs-chain] polaris proc gone" >> $LOG; break; fi
  sleep 60
done

# 2. CheMeleon FROZEN fingerprint probe on CBS (chemeleon venv: chemprop fp + transformers head)
~/venvs/chemeleon/bin/python eval_v2.py --featurizer chemeleon --head mlp --standardize zscore \
  --cv_folds 5 --cv_scheme provided --task_csv data/cbs.csv --task_name cbs --task_type classification \
  --head_seeds 0 1 2 --output_dir figure_data/cbs_benchmark/chemeleon_frozen/moleculenet_cv >> $LOG 2>&1
aws s3 cp --recursive figure_data/cbs_benchmark/chemeleon_frozen/moleculenet_cv \
  s3://climb-s3-bucket/experiments/cbs_benchmark/chemeleon_frozen/moleculenet_cv --only-show-errors >> $LOG 2>&1
echo "[cbs-chain] frozen done $(date -u +%FT%TZ)" >> $LOG

# 3. chemprop END-TO-END arms (vanilla + CheMeleon foundation), 3 seeds each, self-syncing to S3
~/venvs/chemeleon/bin/python scripts/cbs_chemprop_e2e.py >> $LOG 2>&1
echo "[cbs-chain] CBS_ALL_DONE $(date -u +%FT%TZ)" >> $LOG

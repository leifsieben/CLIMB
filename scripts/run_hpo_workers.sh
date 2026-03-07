#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'EOF'
Usage:
  scripts/run_hpo_workers.sh \
    <gpus_csv> <config> <tokenizer_dir> <train_data> <output_dir> <n_trials_per_worker> \
    [extra hyperparameter_search args...]

Example:
  scripts/run_hpo_workers.sh \
    0,1,2,3 \
    configs/config_hyperopt_unsup.yaml \
    /home/ec2-user/artifacts/tokenizer_10 \
    /home/ec2-user/artifacts/train_tokens \
    /home/ec2-user/artifacts/hpo_run_2026_03_07 \
    40 \
    --study_name mlm_hpo_wide --load_if_exists --pruner hyperband --bf16
EOF
  exit 1
fi

gpus_csv="$1"
config="$2"
tokenizer_dir="$3"
train_data="$4"
output_dir="$5"
n_trials_per_worker="$6"
shift 6

mkdir -p "$output_dir" "$output_dir/logs"

IFS=',' read -r -a gpus <<< "$gpus_csv"

for i in "${!gpus[@]}"; do
  gpu="${gpus[$i]}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$output_dir/logs/worker_${i}_gpu${gpu}_${ts}.log"
  out_file="$output_dir/logs/worker_${i}_gpu${gpu}_${ts}.out"

  echo "Starting worker $i on GPU $gpu"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python hyperparameter_search.py \
    --config "$config" \
    --tokenizer "$tokenizer_dir" \
    --train_data "$train_data" \
    --output "$output_dir" \
    --n_trials "$n_trials_per_worker" \
    --worker_id "worker-${i}-gpu-${gpu}" \
    --log_file "$log_file" \
    "$@" \
    > "$out_file" 2>&1 &

  echo "worker $i pid=$!"
done

echo "All workers started."

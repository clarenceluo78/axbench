#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $gpu_count"

# torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
#   --config /home/hl678/concept_llm/axbench/axbench/sweep/haoyan/2b/l20/no_grad.yaml \
#   --dump_dir axbench/results/prod_2b_l20_concept10_no_grad \
#   --overwrite_data_dir axbench/concept10/prod_2b_l20_v1/generate \
#   --run_name official

torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config /home/hl678/concept_llm/axbench/axbench/sweep/haoyan/2b/l20/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l20_concept10_no_grad \
  --overwrite_metadata_dir axbench/concept10/prod_2b_l20_v1/generate \
  --overwrite_inference_data_dir axbench/concept10/prod_2b_l20_v1/inference \
  --run_name official --mode steering

python axbench/scripts/evaluate.py \
  --config /home/hl678/concept_llm/axbench/axbench/sweep/haoyan/2b/l20/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l20_concept10_no_grad \
  --run_name official --mode steering
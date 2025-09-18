#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $gpu_count"

CONFIG_PATH="/home/hl678/concept_llm/axbench/axbench/sweep/haoyan/2b/l20/mat_steer.yaml"
DUMP_DIR="axbench/results/prod_2b_l20_concept10_mat_steer_wosparse"
DATA_DIR="axbench/concept10/prod_2b_l20_v1/generate"
INFER_DATA_DIR="axbench/concept10/prod_2b_l20_v1/inference"

# Training
torchrun --nproc_per_node=${gpu_count} axbench/scripts/train.py \
  --config "${CONFIG_PATH}" \
  --dump_dir "${DUMP_DIR}" \
  --overwrite_data_dir "${DATA_DIR}" \
  --run_name official

# Steering inference
torchrun --nproc_per_node=${gpu_count} axbench/scripts/inference.py \
  --config "${CONFIG_PATH}" \
  --dump_dir "${DUMP_DIR}" \
  --overwrite_metadata_dir "${DATA_DIR}" \
  --overwrite_inference_data_dir "${INFER_DATA_DIR}" \
  --run_name official --mode steering

# # Evaluation
python axbench/scripts/evaluate.py \
  --config "${CONFIG_PATH}" \
  --dump_dir "${DUMP_DIR}" \
  --run_name official --mode steering

#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to the Stable Diffusion 3.5 Medium directory or model id}"
: "${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to the processed five-domain dataset root}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new training-output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
SLRM_GAMMA="${SLRM_GAMMA:-1.0}"
SLRM_EPSILON="${SLRM_EPSILON:-1e-6}"
SLRM_DMAX="${SLRM_DMAX:-512.0}"
HPCS_POWER="${HPCS_POWER:-1.0}"

accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train.py" \
  --pretrained_model_name_or_path "${BASE_MODEL}" \
  --train_data_dir "${TRAIN_DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@" \
  --conditioning_image_column "mask,width_map,sketch,edge,partial_sketch" \
  --min_conditioning_images 1 \
  --max_conditioning_images 1 \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 48 \
  --learning_rate 5e-6 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --adam_weight_decay 1e-2 \
  --max_sequence_length 512 \
  --dataset_preprocess_batch_size 1 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --checkpointing_steps 500 \
  --validation_steps 500 \
  --seed 2026 \
  --hpcs_anchor_ratio 0.4 \
  --hpcs_eta 0.4 \
  --hpcs_power "${HPCS_POWER}" \
  --slrm_gamma "${SLRM_GAMMA}" \
  --slrm_epsilon "${SLRM_EPSILON}" \
  --slrm_dmax "${SLRM_DMAX}"

#!/bin/bash
# MoCaf-Mamba Training Script
# Usage: bash scripts/train.sh [num_gpus]

set -e

NUM_GPUS=${1:-4}
DATA_DIR=${2:-../data}
OUTPUT_DIR=${3:-./checkpoints}
FOLD=${4:-0}
EPOCHS=${5:-300}
LR=${6:-1e-4}

echo "============================================"
echo "MoCaf-Mamba Training"
echo "============================================"
echo "GPUs:        ${NUM_GPUS}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Fold:        ${FOLD}"
echo "Epochs:      ${EPOCHS}"
echo "LR:          ${LR}"
echo "============================================"

torchrun --nproc_per_node=${NUM_GPUS} train_mamba.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "mocaf_mamba" \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}"

echo "Training completed!"

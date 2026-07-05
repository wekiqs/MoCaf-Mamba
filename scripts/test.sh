#!/bin/bash
# MoCaf-Mamba Evaluation Script
# Usage: bash scripts/test.sh [gpu_id] [data_dir]

set -e

GPU=${1:-0}
DATA_DIR=${2:-../data}
CHECKPOINT=${3:-./checkpoints/mocaf_mamba/0/last.pth}

echo "============================================"
echo "MoCaf-Mamba Evaluation"
echo "============================================"
echo "GPU:         ${GPU}"
echo "Data dir:    ${DATA_DIR}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "============================================"

python test_all.py \
    --gpu "${GPU}" \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${CHECKPOINT}"

echo "Evaluation completed!"

#!/bin/bash
# RLT Experiment Runner - Pen Uncap Task
# Usage: bash experiments/run_experiment.sh <run_name> [overrides...]
#
# Examples:
#   bash experiments/run_experiment.sh baseline --phase 1
#   bash experiments/run_experiment.sh fast_lr --phase 1 --rl-token-lr 2e-4
#   bash experiments/run_experiment.sh low_beta --phase 2 --beta 0.5 --rl-token-checkpoint checkpoints/pen_uncap/rl_token/5000
#   bash experiments/run_experiment.sh full_pipeline --phase both --rl-token-lr 2e-4 --beta 0.5 --utd-ratio 10

set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name> [overrides...]}"
shift

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_ENABLE_HF_TRANSFER=1

LOG_DIR="experiments/logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "RLT Experiment: ${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Log dir: ${LOG_DIR}"
echo "Extra args: $@"
echo "================================================================"

uv run python scripts/train_pen_uncap.py \
    --run-name "${RUN_NAME}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "================================================================"
echo "Experiment ${RUN_NAME} complete. Log: ${LOG_DIR}/train.log"
echo "================================================================"

#!/usr/bin/env bash
# Run all experiment configs in parallel, one per GPU.
# Usage: bash run_all.sh [NUM_GPUS]
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${1:-8}"
gpu=0

for cfg in configs/experiments/*.yaml; do
    echo "GPU $gpu: $cfg"
    CUDA_VISIBLE_DEVICES=$gpu uv run python run_experiment.py --config "$cfg" &
    gpu=$(( (gpu + 1) % NUM_GPUS ))
done
wait
echo "All experiments finished."

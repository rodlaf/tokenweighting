#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <config> <output_root> <log_root> <lock_root>" >&2
  exit 2
fi

CONFIG="$1"
OUTPUT_ROOT="$2"
LOG_ROOT="$3"
LOCK_ROOT="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$(basename "$CONFIG" .yaml)"
RUN_LOG="$LOG_ROOT/${RUN_NAME}.log"
STATUS_FILE="$LOG_ROOT/${RUN_NAME}.status"
GPU_IDS="${GPU_IDS:-1 2 3 4 5 6 7}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$LOCK_ROOT"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== deadline MATH run: $RUN_NAME ==="
echo "started_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "config=$CONFIG"
echo "output_root=$OUTPUT_ROOT"
echo "gpu_ids=$GPU_IDS"
echo "log=$RUN_LOG"

OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
if [ -f "$OUT_DIR/final_eval.json" ]; then
  echo "final_eval already exists; skipping completed run."
  echo "SKIPPED_COMPLETED $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$STATUS_FILE"
  exit 0
fi

acquire_gpu() {
  while :; do
    for gpu in $GPU_IDS; do
      local lock_file="$LOCK_ROOT/gpu_${gpu}.lock"
      exec {lock_fd}>"$lock_file"
      if flock -n "$lock_fd"; then
        GPU_LOCK_FD="$lock_fd"
        GPU_ID="$gpu"
        return 0
      fi
      exec {lock_fd}>&-
    done
    echo "waiting_for_gpu $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    sleep 60
  done
}

acquire_gpu
echo "assigned_gpu=$GPU_ID"
echo "RUNNING gpu=$GPU_ID start=$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$STATUS_FILE"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
EXPERIMENT_OUTPUT_ROOT="$OUTPUT_ROOT" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python run_experiment.py --config "$CONFIG" --output-root "$OUTPUT_ROOT" --resume

echo "DONE gpu=$GPU_ID end=$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$STATUS_FILE"
echo "completed_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

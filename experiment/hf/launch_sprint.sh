#!/usr/bin/env bash
# Launch the NeurIPS sprint: 21 Qwen3-4B-Base + POLARIS-53k runs across N GPUs.
#
# Usage:
#   bash launch_sprint.sh                 # uses NUM_GPUS=8 by default
#   NUM_GPUS=4 bash launch_sprint.sh      # use 4 GPUs
#   bash launch_sprint.sh --resume        # only run jobs without final_eval.json
#
# Architecture:
#   * One worker per GPU. Workers race for jobs via a shared file-locked queue.
#   * Each job runs `uv run python run_experiment.py --config <yaml>`.
#   * With --resume, incomplete jobs continue from their latest checkpoint.
#   * Per-job stdout/stderr is captured in sprint_logs/<name>.log.
#   * Workers always pick the next available config; no static sharding so a
#     fast GPU keeps pulling work while a slow one finishes its current job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=${NUM_GPUS:-8}
LOG_DIR="$SCRIPT_DIR/sprint_logs"
mkdir -p "$LOG_DIR"

QUEUE="$LOG_DIR/queue.txt"
LOCK="$LOG_DIR/queue.lock"

RESUME=0
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

# Build the queue.
> "$QUEUE"
for cfg in "$SCRIPT_DIR"/configs/experiments/qwen3-4b-base-polaris-*.yaml; do
  name=$(basename "$cfg" .yaml)
  out_dir="$SCRIPT_DIR/../../outputs/$name"
  if [ "$RESUME" = "1" ] && [ -f "$out_dir/final_eval.json" ]; then
    echo "[resume] skipping completed: $name"
    continue
  fi
  echo "$cfg" >> "$QUEUE"
done

NUM_JOBS=$(wc -l < "$QUEUE" | tr -d ' ')
echo "Queued $NUM_JOBS job(s) across $NUM_GPUS GPU(s). Logs: $LOG_DIR"

if [ "$NUM_JOBS" = "0" ]; then
  echo "Nothing to do."
  exit 0
fi

# Atomically pop a config from the head of the queue. Echoes the path or empty.
pop_config() {
  (
    flock -x 200
    if [ -s "$QUEUE" ]; then
      head -n 1 "$QUEUE"
      sed -i '1d' "$QUEUE"
    fi
  ) 200>"$LOCK"
}

run_worker() {
  local gpu=$1
  while :; do
    cfg=$(pop_config)
    [ -z "$cfg" ] && break
    local name
    name=$(basename "$cfg" .yaml)
    local log="$LOG_DIR/${name}.log"
    local started
    started=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[gpu=$gpu] $started START $name"
    local resume_args=()
    if [ "$RESUME" = "1" ]; then
      resume_args=(--resume)
    fi
    {
      echo "=== launch_sprint.sh: $name on GPU $gpu (UTC $started) ==="
      CUDA_VISIBLE_DEVICES=$gpu \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python run_experiment.py --config "$cfg" "${resume_args[@]}"
    } > "$log" 2>&1 || {
      local ec=$?
      echo "[gpu=$gpu] FAIL $name (exit=$ec); see $log" >&2
      continue
    }
    local ended
    ended=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[gpu=$gpu] $ended DONE  $name"
  done
  echo "[gpu=$gpu] worker exiting"
}

for gpu in $(seq 0 $((NUM_GPUS-1))); do
  run_worker "$gpu" &
done
wait
echo "All sprint jobs complete."

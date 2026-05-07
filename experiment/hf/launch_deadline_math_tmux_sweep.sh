#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-qwen3_17b_math_deadline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs_qwen3_17b_math_deadline}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/sprint_logs_qwen3_17b_math_deadline}"
LOCK_ROOT="${LOCK_ROOT:-$LOG_ROOT/gpu_locks}"
GPU_IDS="${GPU_IDS:-1 2 3 4 5 6 7}"
RUNNER="$SCRIPT_DIR/run_tmux_deadline_math_job.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; refusing to create duplicate launch." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$LOCK_ROOT"
chmod +x "$RUNNER"

mapfile -t CONFIGS < <(printf '%s\n' "$SCRIPT_DIR"/configs/experiments/qwen3-1.7b-base-math-deadline-*.yaml | sort)
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "no deadline MATH configs found" >&2
  exit 1
fi

MANIFEST="$LOG_ROOT/manifest.txt"
{
  echo "session=$SESSION"
  echo "started_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "output_root=$OUTPUT_ROOT"
  echo "log_root=$LOG_ROOT"
  echo "lock_root=$LOCK_ROOT"
  echo "gpu_ids=$GPU_IDS"
  echo "num_configs=${#CONFIGS[@]}"
  printf '%s\n' "${CONFIGS[@]}"
} > "$MANIFEST"

first=1
idx=0
for cfg in "${CONFIGS[@]}"; do
  idx=$((idx + 1))
  name="$(basename "$cfg" .yaml)"
  win="$(printf '%02d-%s' "$idx" "$name")"
  win="${win:0:80}"
  cmd="GPU_IDS='$GPU_IDS' bash '$RUNNER' '$cfg' '$OUTPUT_ROOT' '$LOG_ROOT' '$LOCK_ROOT'"
  if [ "$first" -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$win" "$cmd"
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
    first=0
  else
    tmux new-window -t "$SESSION" -n "$win" "$cmd"
  fi
done

echo "Launched ${#CONFIGS[@]} tmux windows in session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo "Logs: $LOG_ROOT"
echo "Outputs: $OUTPUT_ROOT"

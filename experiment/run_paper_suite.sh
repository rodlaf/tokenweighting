#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_uniform.yaml
python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_surprisal.yaml
python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_entropy.yaml
python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_uniform.yaml
python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_surprisal.yaml
python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_entropy.yaml

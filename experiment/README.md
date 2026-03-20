
# Experiment Suite

This directory contains a lightweight, fully runnable reference implementation for intrinsic token weighting under sparse sequence-level rewards.

What is included:
- a small autoregressive contextual policy implemented in NumPy
- verifiable synthetic arithmetic and program-trace tasks
- RLOO and GRPO-style sequence baselines
- token weighting modes: uniform, surprisal, entropy_reduction
- YAML configs for smoke tests and paper-scale controlled studies

Why this design:
- It runs on CPU with only NumPy + PyYAML.
- It isolates token-level credit assignment without requiring a heavyweight LLM stack.
- It produces exact verifiable rewards, token-weight diagnostics, and seed-aggregated summaries.

## Install

From the repo root:

python3 -m pip install -r experiment/requirements.txt

## Run a smoke experiment

python3 experiment/run_experiment.py --config experiment/configs/smoke_arithmetic_uniform.yaml

## Run the controlled paper suite

Arithmetic (RLOO):
- python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_uniform.yaml
- python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_surprisal.yaml
- python3 experiment/run_experiment.py --config experiment/configs/paper_arithmetic_rloo_entropy.yaml

Program trace (GRPO):
- python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_uniform.yaml
- python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_surprisal.yaml
- python3 experiment/run_experiment.py --config experiment/configs/paper_program_grpo_entropy.yaml

Results are written to experiment/results/<config-name>-<timestamp>/ with per-seed summaries and an aggregate_summary.json.

To turn multiple aggregate summaries into a markdown table:

python3 experiment/report_results.py \
  experiment/results/<run-a>/aggregate_summary.json \
  experiment/results/<run-b>/aggregate_summary.json

## Task definitions

### arithmetic_trace
Prompt: two integers a and b in [0, 19].
Generated sequence: [style, plan, carry, tens, ones].
Reward: 1 iff the carry token and both answer digits are correct.
Important positions: carry, tens, ones.

### program_trace
Prompt: a symbolic code operation plus a small value bucket.
Generated sequence: [style, plan, op, arg, out].
Reward: 1 iff the executable symbolic trace matches the verifier's expected op/arg/out triple.
Important positions: op, arg, out.

## Metrics

- greedy_accuracy: exact reward under greedy decoding
- pass_at_k: success rate over sampled completions
- important_mass_mean: average fraction of token weight placed on semantically important positions
- final_gradient_norm_variance: weighted within-trajectory variance proxy over token score norms

## Notes

These experiments are intended as a controlled credit-assignment testbed and reference implementation. They are lightweight by design, easy to inspect, and suitable for reproducible ablations.

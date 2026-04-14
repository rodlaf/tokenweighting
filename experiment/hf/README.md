# Intrinsic Token Weighting -- Experiments

On-policy RL training with pluggable per-token credit redistribution, as described in the paper. Runs locally on a single GPU using PyTorch + LoRA.

## Setup

```bash
cd experiment/hf
uv sync
```

Requires CUDA. Uses 4-bit quantization by default so experiments fit on a single 24 GB GPU.

## Run an experiment

```bash
uv run python run_experiment.py --config configs/experiments/qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml
```

Validate config without training:

```bash
uv run python run_experiment.py --config configs/experiments/qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml --dry-run
```

## Configs

All experiment configs extend a base config and override `algorithm` + `weighting`.

### GSM8K (Qwen2.5-Math-1.5B-Instruct)

| Config | Algorithm | Weighting |
|--------|-----------|-----------|
| `qwen25-math-1.5b-gsm8k-rloo-uniform.yaml` | RLOO | uniform |
| `qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml` | RLOO | surprisal |
| `qwen25-math-1.5b-gsm8k-rloo-entropy.yaml` | RLOO | entropy reduction |
| `qwen25-math-1.5b-gsm8k-grpo-uniform.yaml` | GRPO | uniform |
| `qwen25-math-1.5b-gsm8k-grpo-surprisal.yaml` | GRPO | surprisal |
| `qwen25-math-1.5b-gsm8k-grpo-entropy.yaml` | GRPO | entropy reduction |

### MBPP (Qwen2.5-Coder-1.5B-Instruct)

| Config | Algorithm | Weighting |
|--------|-----------|-----------|
| `qwen25-coder-1.5b-mbpp-rloo-uniform.yaml` | RLOO | uniform |
| `qwen25-coder-1.5b-mbpp-rloo-surprisal.yaml` | RLOO | surprisal |
| `qwen25-coder-1.5b-mbpp-rloo-entropy.yaml` | RLOO | entropy reduction |
| `qwen25-coder-1.5b-mbpp-grpo-uniform.yaml` | GRPO | uniform |
| `qwen25-coder-1.5b-mbpp-grpo-surprisal.yaml` | GRPO | surprisal |
| `qwen25-coder-1.5b-mbpp-grpo-entropy.yaml` | GRPO | entropy reduction |

## Project structure

```
├── run_experiment.py    # Training loop and CLI entrypoint
├── data.py              # Dataset loading and prompt formatting
├── rewards.py           # Reward dispatch (GSM8K / MBPP)
├── token_weights.py     # Weighting: uniform, surprisal, entropy reduction
├── tasks/
│   ├── gsm8k.py         # Answer extraction and exact-match grading
│   └── mbpp.py          # Code extraction and sandboxed test execution
└── configs/
    ├── base/            # Shared model + training hyperparameters
    └── experiments/     # One YAML per (model, task, algorithm, weighting)
```

## Adding a new weighting method

1. Add a weight-computation function in `token_weights.py`
2. Register it in `build_token_weights`
3. Add the name to the validation check in `run_experiment.py:validate_config`

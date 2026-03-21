# tokenweighting

Intrinsic token weighting (ITW) experiments for critic-free RL on language models.

This branch implements the paper's actual experiment scaffolding rather than a synthetic benchmark:
- Qwen/Qwen2.5-Math-1.5B-Instruct on GSM8K
- Qwen/Qwen2.5-Coder-1.5B-Instruct on MBPP
- RLOO and GRPO baselines
- uniform, surprisal, and entropy-reduction token weighting

Main entrypoint
- `python experiment/run_experiment.py --config <config.yaml>`

Useful configs
- `configs/experiments/qwen25-math-1.5b-gsm8k-rloo-uniform.yaml`
- `configs/experiments/qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml`
- `configs/experiments/qwen25-math-1.5b-gsm8k-rloo-entropy.yaml`
- `configs/experiments/qwen25-math-1.5b-gsm8k-grpo-uniform.yaml`
- `configs/experiments/qwen25-math-1.5b-gsm8k-grpo-surprisal.yaml`
- `configs/experiments/qwen25-math-1.5b-gsm8k-grpo-entropy.yaml`
- `configs/experiments/qwen25-coder-1.5b-mbpp-rloo-uniform.yaml`
- `configs/experiments/qwen25-coder-1.5b-mbpp-rloo-surprisal.yaml`
- `configs/experiments/qwen25-coder-1.5b-mbpp-rloo-entropy.yaml`

See `experiment/README.md` for install and run instructions.

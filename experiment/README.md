# ITW Experiment Suite

This repository now contains real experiment code for the paper's Qwen-based setup.

Implemented tasks
- GSM8K with exact-answer binary rewards
- MBPP with unit-test pass/fail rewards

Implemented methods
- RLOO + uniform token weighting
- RLOO + surprisal weighting
- RLOO + entropy-reduction weighting
- GRPO + uniform token weighting
- GRPO + surprisal weighting
- GRPO + entropy-reduction weighting

Implemented models/configs
- Qwen/Qwen2.5-Math-1.5B-Instruct on GSM8K
- Qwen/Qwen2.5-Coder-1.5B-Instruct on MBPP

Install
```bash
python -m pip install -r experiment/requirements.txt
```

Validate a config without training
```bash
python experiment/run_experiment.py --config configs/experiments/qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml --dry-run
```

Run an experiment
```bash
python experiment/run_experiment.py --config configs/experiments/qwen25-math-1.5b-gsm8k-rloo-surprisal.yaml
```

Notes
- The code intentionally does not ship precomputed results.
- The user should run the experiments and populate paper numbers from actual outputs.
- MBPP execution uses a lightweight timeout/process-isolation harness; for hardened sandboxing, run inside a container.

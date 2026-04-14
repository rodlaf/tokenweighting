# Token-Weighting Experiments (Tinker)

Experiment infrastructure for token-level credit assignment in language RL, built on [Tinker](https://tinker-docs.thinkingmachines.ai/).

## Framework

All methods decompose per-token credit as:

```
c_t = alpha_t * A + psi_t
```

where `alpha_t` redistributes the trajectory-level advantage and `psi_t` injects an additive process signal.

### Implemented methods

| Method | `--alpha` | `--psi` | Reference |
|--------|-----------|---------|-----------|
| Standard GRPO/RLOO | `uniform` | `none` | [DeepSeekMath (Shao+ 2024)](https://arxiv.org/abs/2402.03300) |
| ITW-Surprisal | `surprisal` | `none` | Ours |
| ITW-Entropy-Reduction | `entropy_reduction` | `none` | Ours |
| Entropy Magnitude | `entropy_magnitude` | `none` | -- |
| Top-K Mask (80/20) | `topk_mask` | `none` | [Wang+ 2025](https://arxiv.org/abs/2506.01939) |
| REPO-R | `uniform` | `repo_rescale` | [Petrenko+ 2026](https://arxiv.org/abs/2603.11682) |
| ERPO | `erpo_gating` | `erpo_progress` | [Yu+ 2026](https://arxiv.org/abs/2603.28204) |
| Surprisal + REPO (novel) | `surprisal` | `repo_rescale` | Ours |
| Entropy-Red + REPO (novel) | `entropy_reduction` | `repo_rescale` | Ours |

### Supported base algorithms

| `--algorithm` | Loss | Notes |
|---------------|------|-------|
| `grpo` | importance sampling | Default, group-relative z-scored advantages |
| `rloo` | importance sampling | Leave-one-out baseline |
| `dapo` | PPO (asymmetric clip 0.8/1.28) | Stronger baseline, addresses entropy collapse |
| `cispo` | CISPO | Clipped ratio as gradient coefficient |

## Quick start

```bash
cd experiment/tinker
uv sync

# Validate the full pipeline
tw smoke

# Train from a config
tw train --config configs/gsm8k-surprisal.yaml

# Train from CLI flags
tw train --algorithm dapo --alpha entropy_reduction --model Qwen/Qwen3-8B --steps 200

# Novel combination: both axes
tw train --alpha surprisal --psi repo_rescale --model Qwen/Qwen3-8B

# Compare runs
tw plot outputs/run1 outputs/run2 outputs/run3
tw plot outputs/* --metric batch_mean_entropy --save entropy.png
```

## CLI reference

```
tw train [OPTIONS]
  --config PATH           YAML config file
  --model TEXT             Model name (overrides config)
  --algorithm [grpo|rloo|dapo|cispo]
  --alpha [uniform|surprisal|entropy_reduction|entropy_magnitude|topk_mask|erpo_gating]
  --psi [none|centered_logprob|repo_rescale|erpo_progress]
  --steps INTEGER
  --lr FLOAT
  --group-size INTEGER
  --output TEXT
  --wandb TEXT             W&B project name (enables logging)

tw plot RUN_DIRS...
  --metric TEXT            Metric to plot (default: reward_mean)
  --smooth INTEGER         EMA smoothing span (default: 5)
  --save TEXT              Save to file instead of showing

tw smoke                   End-to-end pipeline smoke test
```

## Project structure

```
├── cli.py              # Entrypoint (installed as `tw`)
├── configs/            # YAML experiment configs
├── src/
│   ├── credit.py       # c_t = alpha * A + psi framework
│   ├── entropy.py      # Top-k logprobs -> entropy estimation
│   └── scripts/
│       ├── grpo.py     # Training loop
│       └── plot.py     # Metric visualization
└── tests/
    └── smoke_test.py
```

## Adding a new method

1. Add a class to `src/credit.py` implementing `MultiplicativeWeight` or `AdditiveSignal`
2. Register it in the `ALPHAS` or `PSIS` dict
3. Create a config in `configs/` (or just use `--alpha`/`--psi` CLI flags)

## SkyRL compatibility

This codebase targets the Tinker API. [SkyRL](https://docs.skyrl.ai/docs/tinker/overview) implements the same API on private infrastructure (FSDP2/Megatron backends), enabling zero-code-change migration to your own GPUs.

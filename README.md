# ARCA

Code for the ARCA paper: adapter-residual token credit assignment for LoRA-based
LLM reinforcement learning.

The reported sweep fine-tunes `Qwen/Qwen3-1.7B-Base` on MATH with GRPO, binary
exact-match rewards, and LoRA. It compares uniform, surprisal,
entropy-reduction, policy-divergence, and adapter-residual (ARCA) token
weighting.

```bash
cd experiment/hf
uv sync
CUDA_VISIBLE_DEVICES=0 uv run python run_experiment.py \
  --config configs/experiments/qwen3-1.7b-base-math-deadline-grpo-adapter-residual-r64-s1337.yaml
```

The seven paper configs live under `experiment/hf/configs/experiments/` as
`qwen3-1.7b-base-math-deadline-grpo-*.yaml`.

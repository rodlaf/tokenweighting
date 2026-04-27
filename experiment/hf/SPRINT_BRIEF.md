# NeurIPS Sprint Brief — Qwen3-4B-Base on POLARIS-53k

## TL;DR for the runner

Pull the repo, `cd experiment/hf`, then on a node with 8x H100 80GB:

```bash
bash launch_sprint.sh
```

That kicks off 21 RL training runs across the 8 GPUs with a work-stealing
queue. Each run is fully self-contained and writes to its own output directory
under `outputs/<run_name>/`. Per-job logs live in `experiment/hf/sprint_logs/`.

Re-run with `bash launch_sprint.sh --resume` to skip any runs that already
have `final_eval.json` (idempotent).

## What we are testing

**Hypothesis.** Heuristic token-weighting (surprisal, divergence) collapses
under LoRA because the LoRA delta is concentrated on a small fraction of
tokens, breaking token-level credit assignment. Our proposed method —
**Adapter-Residual Token Weighting (ARTW)** — uses the magnitude of the LoRA
adapter's residual contribution itself as the per-token salience signal,
making the weighting consistent with where the policy actually changes.

We expect ARTW to (a) match or beat uniform on reward, (b) crush the
heuristics on token-weight concentration metrics (Gini, effective number of
tokens), and (c) hold up under a rank ablation and an algorithm ablation.

## Setup

- **Model:** `Qwen/Qwen3-4B-Base` (no instruct tuning).
- **Adapter:** LoRA `r=64, alpha=128` for headlines (`r=4`, `r=16` ablations).
- **Training data:** `POLARIS-Project/Polaris-Dataset-53K` (52,291 problems
  after holding out 1k for eval). Split is deterministic — same eval set
  across all runs/seeds (see `data.py::POLARIS_SPLIT_SEED`).
- **Algorithm:** RLOO for headlines, GRPO for the algorithm-ablation runs.
- **Hyperparams (`configs/base/qwen3-4b-base-polaris.yaml`):**
  `batch_size=1, grad_accum=16, num_generations=8, max_new_tokens=1024,
   T=1.0, top_p=0.95, lr=5e-6, warmup=40, train_steps=400`.
  Effective batch = 16 prompts/step, 128 rollouts/step.
- **Reward:** Verifiable answer match — extract `\boxed{...}` from the
  completion and compare to the gold POLARIS answer (`tasks/polaris.py`).

## The 21 runs

All configs are in `experiment/hf/configs/experiments/qwen3-4b-base-polaris-*.yaml`.
Three seeds: `1337`, `2024`, `3141`.

### A. Headline (12) — 4 methods x 3 seeds, RLOO, LoRA r=64

| weighting          | files                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| `uniform`          | `qwen3-4b-base-polaris-rloo-uniform-s{1337,2024,3141}.yaml`                 |
| `surprisal`        | `qwen3-4b-base-polaris-rloo-surprisal-s{1337,2024,3141}.yaml`               |
| `divergence`       | `qwen3-4b-base-polaris-rloo-divergence-s{1337,2024,3141}.yaml`              |
| `adapter_residual` | `qwen3-4b-base-polaris-rloo-adapter-residual-r64-s{1337,2024,3141}.yaml`    |

### B. Rank ablation (6) — ARTW × {r=4, r=16} × 3 seeds

| rank | files                                                                       |
| ---- | --------------------------------------------------------------------------- |
| 4    | `qwen3-4b-base-polaris-rloo-adapter-residual-r4-s{1337,2024,3141}.yaml`     |
| 16   | `qwen3-4b-base-polaris-rloo-adapter-residual-r16-s{1337,2024,3141}.yaml`    |

(The r=64 row from the headline category completes this ablation.)

### C. Algorithm ablation (3) — GRPO + ARTW r=64 × 3 seeds

`qwen3-4b-base-polaris-grpo-adapter-residual-r64-s{1337,2024,3141}.yaml`

## Evaluation

For **every** run, after training finishes the harness writes:

- `final_eval.json` — primary eval on **POLARIS-eval (1,000 held-out problems)**.
  This is the headline number for the paper.
- `final_eval_math.json` — bonus OOD eval on **MATH-test (500)**.
- `final_eval_gsm8k.json` — bonus OOD eval on **GSM8K-test (200)**.

Mid-training eval on POLARIS-eval also runs every 50 steps
(`eval_step_50.json`, …, `eval_step_400.json`) for learning curves.
Per-step training metrics (reward, loss, weight Gini, effective number of
tokens, …) are appended to `train_metrics.jsonl`.

## Resource expectations

- **VRAM:** ~70 GB peak per H100 with `batch_size=1, max_new=1024`. The
  `divergence` runs additionally hold a frozen reference model when LoRA is
  off, but with LoRA enabled (our case) we use the base weights as the
  reference and stay at one model in VRAM.
- **Per-run wall time:** ~14–20 hours on a single H100. Bonus evals add
  ~30–60 min at the end.
- **Total wall time:** With dynamic queueing across 8 GPUs and 21 jobs of
  similar cost, expect the sprint to finish in **~2.5–3 days**.

## Sanity checks before walking away

After roughly 30 minutes per run, peek at the first few lines of
`train_metrics.jsonl` for a couple of jobs:

```bash
cd experiment/hf
for d in ../../outputs/qwen3-4b-base-polaris-rloo-uniform-s1337 \
         ../../outputs/qwen3-4b-base-polaris-rloo-adapter-residual-r64-s1337; do
  echo "=== $d ==="
  head -n 3 "$d/train_metrics.jsonl"
done
```

You should see:
- `reward_mean` non-zero (POLARIS is solvable for Qwen3-4B-Base ~10–25 % at
  step 1; if it is 0 across all runs, the prompt or reward extraction is
  broken).
- `weight_gini` and `effective_num_tokens` populated.
- Reward trending **up** (not flat) by step ~100 for at least the
  `adapter_residual` and `uniform` runs.

If a run OOMs, the most likely culprits are:
1. Another process holding VRAM on that GPU — check `nvidia-smi`.
2. Tokenizer/special-token mismatch causing very long completions — confirm
   `max_new_tokens=1024` is being respected in the logs.

If everything looks healthy, just let it cook.

## Hand-off when complete

Once `launch_sprint.sh` returns, ping me with:
1. The contents of `sprint_logs/` (or just confirmation every job exited 0).
2. The list of populated `outputs/qwen3-4b-base-polaris-*/final_eval*.json`
   files.

I'll handle figure generation and table population from there.

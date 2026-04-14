"""
GRPO training with pluggable credit assignment on Tinker.

Supports all credit methods (uniform, surprisal, entropy-reduction, REPO, ERPO)
via the credit framework. Called by cli.py.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import tinker
import torch
from tinker import TensorData

from tinker_cookbook.renderers import get_renderer, get_text_content

from src.credit import (
    AdaptiveZetaController,
    CreditFunction,
    ERPOGating,
    ERPOProgress,
    REPORescale,
    TokenSignals,
    build_credit_function,
)
from src.entropy import compute_entropies, get_topk_logprobs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    import yaml

    cfg_path = Path(path)
    config = yaml.safe_load(cfg_path.read_text())
    base = config.pop("base_config", None)
    if base:
        base_path = (cfg_path.parent / base).resolve()
        parent = load_config(str(base_path))
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(parent.get(k), dict):
                parent[k] = {**parent[k], **v}
            else:
                parent[k] = v
        config = parent
    return config


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    return match[-1].strip() if match else None


def grade_gsm8k(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if answer.replace(",", "").strip() == ground_truth.replace(",", "").strip() else 0.0


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    raise ValueError("No #### answer found")


# ---------------------------------------------------------------------------
# Feature detection from credit function
# ---------------------------------------------------------------------------


def _needs_entropies(credit_fn: CreditFunction) -> bool:
    """Does this credit function need per-token entropy estimates?"""
    from src.credit import EntropyMagnitude, EntropyReduction, TopKMask

    entropy_alphas = (EntropyReduction, EntropyMagnitude, TopKMask, ERPOGating)
    entropy_psis = (REPORescale,)
    return isinstance(credit_fn.alpha, entropy_alphas) or isinstance(credit_fn.psi, entropy_psis)


def _needs_reference(credit_fn: CreditFunction) -> bool:
    """Does this credit function need reference policy logprobs?"""
    return isinstance(credit_fn.psi, ERPOProgress)


def _needs_adaptive_zeta(config: dict[str, Any]) -> bool:
    return config.get("training", {}).get("adaptive_zeta", False)


# ---------------------------------------------------------------------------
# Base algorithm: advantage computation + loss config
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "grpo": {
        "loss_fn": "importance_sampling",
        "loss_fn_config": {},
    },
    "rloo": {
        "loss_fn": "importance_sampling",
        "loss_fn_config": {},
    },
    "dapo": {
        "loss_fn": "ppo",
        "loss_fn_config": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
    },
    "cispo": {
        "loss_fn": "cispo",
        "loss_fn_config": {},
    },
}


def _compute_advantages(
    rewards: list[float],
    algorithm: str,
    eps: float = 1e-8,
) -> list[float]:
    """Compute per-trajectory advantages within a group.

    Args:
        rewards: Per-trajectory rewards for one prompt group.
        algorithm: One of grpo, rloo, dapo, cispo.

    Returns:
        Per-trajectory advantage values.
    """
    n = len(rewards)
    mean_r = sum(rewards) / n

    if algorithm == "rloo":
        # Leave-one-out baseline: A_i = r_i - mean(r_{j != i})
        total = sum(rewards)
        return [r - (total - r) / max(n - 1, 1) for r in rewards]

    # GRPO / DAPO / CISPO: z-scored group advantages
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / n) ** 0.5
    if std_r < eps:
        return [0.0] * n
    return [(r - mean_r) / (std_r + eps) for r in rewards]


# ---------------------------------------------------------------------------
# ERPO cross-group preprocessing
# ---------------------------------------------------------------------------


def _bucket_normalize_progress(
    progress_all: list[torch.Tensor],
    masks: list[torch.Tensor],
    n_buckets: int = 10,
    eps: float = 1e-8,
) -> list[torch.Tensor]:
    """Z-score normalize progress signals within temporal buckets across the group.

    ERPO Eq 9: partition each trajectory into K positional buckets by relative
    position tau = t / T, then z-score within each bucket across all trajectories
    in the group.
    """
    # Collect (bucket_id, value) pairs across all trajectories
    bucket_values: dict[int, list[float]] = {k: [] for k in range(n_buckets)}
    bucket_indices: list[list[tuple[int, int]]] = []  # per-traj list of (position, bucket)

    for i, (prog, mask) in enumerate(zip(progress_all, masks)):
        T = int(mask.sum().item())
        traj_buckets = []
        for t in range(prog.shape[0]):
            if mask[t] > 0 and T > 0:
                bucket = min(int(t / T * n_buckets), n_buckets - 1)
                bucket_values[bucket].append(prog[t].item())
                traj_buckets.append((t, bucket))
        bucket_indices.append(traj_buckets)

    # Compute per-bucket stats
    bucket_mean: dict[int, float] = {}
    bucket_std: dict[int, float] = {}
    for k, vals in bucket_values.items():
        if vals:
            t = torch.tensor(vals)
            bucket_mean[k] = t.mean().item()
            bucket_std[k] = t.std().item() + eps
        else:
            bucket_mean[k] = 0.0
            bucket_std[k] = 1.0

    # Z-score each trajectory's progress signal
    normalized = []
    for i, (prog, mask) in enumerate(zip(progress_all, masks)):
        normed = torch.zeros_like(prog)
        for t, bucket in bucket_indices[i]:
            normed[t] = (prog[t].item() - bucket_mean[bucket]) / bucket_std[bucket]
        normalized.append(normed * mask)

    return normalized


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def run_training(config: dict[str, Any]) -> None:
    # -- Unpack config --
    model_name = config["model"]["name"]
    renderer_name = config["model"].get("renderer", "qwen3")
    lora_rank = config["model"].get("lora_rank", 32)
    use_reference = config["model"].get("reference", False)
    lr = config["training"]["learning_rate"]
    n_steps = config["training"]["train_steps"]
    batch_size = config["training"]["batch_size"]
    group_size = config["training"]["group_size"]
    max_tokens = config["training"]["max_new_tokens"]
    temperature = config["training"].get("temperature", 0.8)
    topk_entropy = config["training"].get("topk_entropy", 50)
    algorithm = config["training"].get("algorithm", "grpo")
    adaptive_zeta = _needs_adaptive_zeta(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_cfg = ALGORITHMS[algorithm]
    loss_fn = algo_cfg["loss_fn"]
    loss_fn_config = algo_cfg["loss_fn_config"] or None

    # -- W&B logging --
    wandb_project = config.get("wandb_project")
    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=config.get("wandb_name") or output_dir.name,
            config=config,
        )

    # -- Build credit function --
    credit_cfg = config.get("credit", {})
    credit_fn = build_credit_function(
        alpha=credit_cfg.get("alpha", "uniform"),
        psi=credit_cfg.get("psi", "none"),
        alpha_kwargs=credit_cfg.get("alpha_kwargs"),
        psi_kwargs=credit_cfg.get("psi_kwargs"),
    )

    want_entropies = _needs_entropies(credit_fn)
    want_reference = _needs_reference(credit_fn) or use_reference

    # -- Tinker clients --
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name, rank=lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)
    adam_params = tinker.AdamParams(learning_rate=lr)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    # Reference policy client (frozen base model, for ERPO)
    reference_client: tinker.SamplingClient | None = None
    if want_reference:
        reference_client = service_client.create_sampling_client(base_model=model_name)
        print(f"Created reference policy client for {model_name}")

    # Adaptive zeta controller (for REPO-R)
    zeta_controller: AdaptiveZetaController | None = None
    if adaptive_zeta:
        initial_zeta = credit_cfg.get("psi_kwargs", {}).get("zeta", 1.0)
        zeta_controller = AdaptiveZetaController(initial_zeta=initial_zeta)

    # -- Dataset --
    import datasets
    ds = datasets.load_dataset("openai/gsm8k", "main")
    train_data = ds["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    fewshot = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {"role": "assistant", "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        )},
    ]

    # -- Save resolved config --
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics_path = output_dir / "train_metrics.jsonl"

    for step in range(n_steps):
        step_start = time.time()
        batch_entropies: list[float] = []  # for adaptive zeta

        # 1. Sample a batch of problems
        start_idx = (step * batch_size) % len(train_data)
        batch = train_data.select(range(start_idx, min(start_idx + batch_size, len(train_data))))

        # 2. Get current policy for sampling
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # 3. Generate rollouts
        prompts = []
        sample_coros = []
        for question in batch["question"]:
            convo = [*fewshot, {"role": "user", "content": question + question_suffix}]
            prompt = renderer.build_generation_prompt(convo)
            prompts.append(prompt)
            sample_coros.append(
                sampling_client.sample_async(
                    prompt=prompt, num_samples=group_size, sampling_params=sampling_params,
                )
            )
        sample_results = await asyncio.gather(*sample_coros)

        # 4. Grade, compute advantages, build datums with credit assignment
        datums: list[tinker.Datum] = []
        all_rewards: list[float] = []
        n_degenerate = 0

        for prob_idx, (sample_result, prompt, answer_text) in enumerate(
            zip(sample_results, prompts, batch["answer"])
        ):
            ground_truth = extract_gsm8k_answer(answer_text)
            prompt_len = prompt.length

            # Collect rollout data per group member
            rewards_G: list[float] = []
            tokens_G: list[list[int]] = []
            logprobs_G: list[list[float]] = []

            for seq in sample_result.sequences:
                tokens_G.append(seq.tokens)
                logprobs_G.append(seq.logprobs)
                parsed, _ = renderer.parse_response(seq.tokens)
                reward = grade_gsm8k(get_text_content(parsed), ground_truth)
                rewards_G.append(reward)

            # Group-relative advantages (GRPO/RLOO/DAPO/CISPO)
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = _compute_advantages(rewards_G, algorithm)
            all_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                n_degenerate += 1
                continue

            # -- Optional: top-k logprobs for entropy estimation --
            entropies_G: list[torch.Tensor | None] = [None] * len(tokens_G)
            if want_entropies:
                full_seqs = [
                    prompt.append(tinker.EncodedTextChunk(tokens=toks))
                    for toks in tokens_G
                ]
                topk_results = await get_topk_logprobs(
                    sampling_client, full_seqs, topk=topk_entropy,
                )
                entropies_G = [
                    compute_entropies(tk, start=prompt_len, length=len(toks))
                    for tk, toks in zip(topk_results, tokens_G)
                ]
                # Track for adaptive zeta
                for ent in entropies_G:
                    if ent is not None:
                        batch_entropies.append(ent.mean().item())

            # -- Optional: reference policy logprobs (for ERPO) --
            ref_logprobs_G: list[torch.Tensor | None] = [None] * len(tokens_G)
            if want_reference and reference_client is not None:
                full_seqs = [
                    prompt.append(tinker.EncodedTextChunk(tokens=toks))
                    for toks in tokens_G
                ]
                ref_lp_results = await asyncio.gather(*[
                    reference_client.compute_logprobs_async(seq)
                    for seq in full_seqs
                ])
                ref_logprobs_G = [
                    torch.tensor(
                        [lp if lp is not None else 0.0 for lp in rlp[prompt_len: prompt_len + len(toks)]],
                        dtype=torch.float32,
                    )
                    for rlp, toks in zip(ref_lp_results, tokens_G)
                ]

            # -- Optional: ERPO cross-group bucket normalization --
            if isinstance(credit_fn.psi, ERPOProgress) and ref_logprobs_G[0] is not None:
                beta_p = credit_fn.psi.beta_progress
                progress_raw = [
                    beta_p * (torch.tensor(lp, dtype=torch.float32) - rlp)
                    for lp, rlp in zip(logprobs_G, ref_logprobs_G)
                    if rlp is not None
                ]
                masks = [torch.ones(len(toks), dtype=torch.float32) for toks in tokens_G]
                progress_normed = _bucket_normalize_progress(progress_raw, masks)
                # Overwrite ref_logprobs with the normalized progress signal.
                # ERPOProgress will read this and apply gating + outcome anchoring.
                # We set ref_logprobs = normed_progress / beta_progress so that
                # when ERPOProgress computes beta*(logprobs - ref_logprobs) it
                # gets the bucket-normalized value directly.
                ref_logprobs_G = [
                    torch.tensor(lp, dtype=torch.float32) - (pn / beta_p)
                    for lp, pn in zip(logprobs_G, progress_normed)
                ]

            # -- Build per-completion datums with credit-weighted advantages --
            ob_len = prompt_len - 1
            for tokens, logprobs, advantage, ent, ref_lp in zip(
                tokens_G, logprobs_G, advantages_G, entropies_G, ref_logprobs_G
            ):
                signals = TokenSignals(
                    logprobs=torch.tensor(logprobs, dtype=torch.float32),
                    mask=torch.ones(len(tokens), dtype=torch.float32),
                    entropies=ent,
                    ref_logprobs=ref_lp,
                )
                credits = credit_fn.compute(signals, advantage)

                model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + credits.tolist()

                datums.append(tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                ))

        # 5. Adaptive zeta update (REPO-R)
        if zeta_controller and batch_entropies:
            new_zeta = zeta_controller.update(sum(batch_entropies) / len(batch_entropies))
            # Reconstruct psi with updated zeta
            if isinstance(credit_fn.psi, REPORescale):
                credit_fn.psi.zeta = new_zeta

        # 6. Train
        if datums:
            fwd_kwargs: dict[str, Any] = {"loss_fn": loss_fn}
            if loss_fn_config:
                fwd_kwargs["loss_fn_config"] = loss_fn_config
            fwd_future = await training_client.forward_backward_async(
                datums, **fwd_kwargs,
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_future.result_async()
            await optim_future.result_async()

        # 7. Log
        record: dict[str, Any] = {
            "step": step,
            "algorithm": algorithm,
            "reward_mean": sum(all_rewards) / max(len(all_rewards), 1),
            "n_datums": len(datums),
            "n_degenerate": n_degenerate,
            "seconds": round(time.time() - step_start, 2),
        }
        if batch_entropies:
            record["batch_mean_entropy"] = round(sum(batch_entropies) / len(batch_entropies), 4)
        if zeta_controller:
            record["zeta"] = round(zeta_controller.zeta, 4)
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if wandb_run:
            wandb_run.log(record, step=step)
        print(json.dumps(record))

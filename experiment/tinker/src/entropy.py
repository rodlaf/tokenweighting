"""
Entropy estimation from Tinker's top-k prompt logprobs API.

After generating a rollout, we pass the full sequence back as a "prompt"
with max_tokens=1 and topk_prompt_logprobs=K. This gives us the model's
top-K next-token distribution at every position, from which we approximate
per-position entropy.

This pattern is validated by the tinker-cookbook SDFT recipe, which uses
the same API for distillation.
"""

from __future__ import annotations

import asyncio

import tinker
import torch


async def get_topk_logprobs(
    sampling_client: tinker.SamplingClient,
    sequences: list[tinker.ModelInput],
    topk: int = 50,
) -> list[list[list[tuple[int, float]]] | None]:
    """Fetch top-k logprobs at every position for each sequence.

    Args:
        sampling_client: Tinker sampling client with current policy weights.
        sequences: Full sequences (prompt + completion) as ModelInputs.
        topk: Number of top tokens to retrieve per position.

    Returns:
        List of topk_prompt_logprobs results, one per sequence.
        Each is a list of (token_id, logprob) pairs per position.
    """
    responses = await asyncio.gather(*[
        sampling_client.sample_async(
            prompt=seq,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk,
        )
        for seq in sequences
    ])
    return [r.topk_prompt_logprobs for r in responses]


def entropy_from_topk_at_position(
    topk_entries: list[tuple[int, float]],
) -> float:
    """Compute entropy from a single position's top-k logprobs.

    Renormalizes the top-k probabilities and computes H = -sum(p * log p).
    With k >= 50, this is a tight approximation of true entropy for most
    positions (the tail contributes negligibly).
    """
    if not topk_entries:
        return 0.0
    logprobs = torch.tensor([lp for _, lp in topk_entries], dtype=torch.float32)
    logprobs -= torch.logsumexp(logprobs, dim=0)  # renormalize over top-k
    probs = logprobs.exp()
    return -(probs * logprobs).sum().item()


def compute_entropies(
    topk_result: list[list[tuple[int, float]]] | None,
    start: int,
    length: int,
) -> torch.Tensor:
    """Extract per-position entropy tensor for a span of completion tokens.

    Args:
        topk_result: Full topk_prompt_logprobs from Tinker for one sequence.
        start: Index of the first completion token in the full sequence.
        length: Number of completion tokens.

    Returns:
        (length,) tensor of per-position entropy values.
    """
    entropies = torch.zeros(length, dtype=torch.float32)
    if topk_result is None:
        return entropies
    for t in range(length):
        pos = start + t
        if pos < len(topk_result) and topk_result[pos] is not None:
            entropies[t] = entropy_from_topk_at_position(topk_result[pos])
    return entropies

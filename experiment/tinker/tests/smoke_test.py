"""
Minimal smoke test for the Tinker + credit framework pipeline.

Validates: API connectivity, sampling, top-k logprobs, entropy estimation,
credit computation, and one training step. Costs < $0.10.
"""

from __future__ import annotations

import asyncio
import os
import sys

import tinker
import torch
from tinker import TensorData

from tinker_cookbook.renderers import get_renderer, get_text_content

from src.credit import (
    CreditFunction,
    EntropyReduction,
    NoAdditive,
    REPORescale,
    Surprisal,
    TokenSignals,
    Uniform,
    build_credit_function,
)
from src.entropy import compute_entropies, get_topk_logprobs


MODEL = "meta-llama/Llama-3.2-1B"
RENDERER = "llama3"


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


async def run():
    print("=" * 60)
    print("Smoke test: Tinker + credit framework")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Connect
    # ---------------------------------------------------------------
    print("\n1. Tinker connectivity")
    assert os.environ.get("TINKER_API_KEY"), "Set TINKER_API_KEY in env"
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=16,
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(RENDERER, tokenizer)
    check("Training client created", True, MODEL)

    # ---------------------------------------------------------------
    # 2. Sample one group of completions
    # ---------------------------------------------------------------
    print("\n2. Sampling")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    prompt_messages = [
        {"role": "user", "content": "What is 2 + 3? Answer inside \\boxed{}."},
    ]
    prompt = renderer.build_generation_prompt(prompt_messages)
    sample_result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=4,
        sampling_params=tinker.SamplingParams(
            max_tokens=128,
            stop=renderer.get_stop_sequences(),
            temperature=0.8,
        ),
    )
    check("Got 4 completions", len(sample_result.sequences) == 4)

    seq0 = sample_result.sequences[0]
    check("Tokens non-empty", len(seq0.tokens) > 0, f"{len(seq0.tokens)} tokens")
    check("Logprobs present", len(seq0.logprobs) == len(seq0.tokens))

    parsed, _ = renderer.parse_response(seq0.tokens)
    text = get_text_content(parsed)
    check("Can parse response", len(text) > 0, f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"')

    # ---------------------------------------------------------------
    # 3. Top-k logprobs (entropy estimation path)
    # ---------------------------------------------------------------
    print("\n3. Top-k logprobs for entropy")
    full_seq = prompt.append(tinker.EncodedTextChunk(tokens=seq0.tokens))
    topk_results = await get_topk_logprobs(sampling_client, [full_seq], topk=50)

    check("topk_results non-None", topk_results[0] is not None)
    prompt_len = prompt.length
    comp_len = len(seq0.tokens)
    entropies = compute_entropies(topk_results[0], start=prompt_len, length=comp_len)
    check("Entropies shape matches", entropies.shape[0] == comp_len, f"shape={entropies.shape}")
    check("Entropies non-negative", (entropies >= 0).all().item())
    check("Entropies non-trivial", entropies.max().item() > 0.1, f"max={entropies.max():.3f}, mean={entropies.mean():.3f}")

    # ---------------------------------------------------------------
    # 4. Credit framework
    # ---------------------------------------------------------------
    print("\n4. Credit framework")
    logprobs_t = torch.tensor(seq0.logprobs, dtype=torch.float32)
    mask_t = torch.ones(comp_len, dtype=torch.float32)

    signals = TokenSignals(logprobs=logprobs_t, mask=mask_t, entropies=entropies)

    for name, alpha in [("uniform", Uniform()), ("surprisal", Surprisal()), ("entropy_reduction", EntropyReduction())]:
        cf = CreditFunction(alpha=alpha, psi=NoAdditive())
        credits = cf.compute(signals, advantage=1.0)
        total = credits.sum().item()
        check(f"{name}: credits sum ~ T", abs(total - comp_len) < 1.0, f"sum={total:.2f}, T={comp_len}")

    # Test build_credit_function factory
    cf2 = build_credit_function("surprisal", "centered_logprob", psi_kwargs={"beta": 0.1})
    credits2 = cf2.compute(signals, advantage=1.0)
    check("Factory construction works", credits2.shape[0] == comp_len)

    # ---------------------------------------------------------------
    # 5. One training step
    # ---------------------------------------------------------------
    print("\n5. Training step")
    ob_len = prompt.length - 1
    tokens = seq0.tokens
    logprobs = seq0.logprobs
    advantage = 1.0

    cf = CreditFunction(alpha=Surprisal(), psi=NoAdditive())
    signals = TokenSignals(
        logprobs=torch.tensor(logprobs, dtype=torch.float32),
        mask=torch.ones(len(tokens), dtype=torch.float32),
        entropies=None,
    )
    credits = cf.compute(signals, advantage)

    model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
    target_tokens = [0] * ob_len + tokens
    padded_logprobs = [0.0] * ob_len + logprobs
    padded_advantages = [0.0] * ob_len + credits.tolist()

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
        },
    )

    fwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    result = await fwd_future.result_async()
    check("forward_backward succeeded", "loss:sum" in result.metrics, f"loss={result.metrics.get('loss:sum', '?')}")

    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=4e-5))
    await optim_future.result_async()
    check("optim_step succeeded", True)

    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from tasks.gsm8k import gsm8k_reward as _gsm8k_reward
from tasks.math_task import math_reward as _math_reward
from tasks.mbpp import estimate_pass_at_k, mbpp_pass_fail_reward


def score_completion(example: dict, completion: str, *, timeout: float = 3.0) -> int:
    task = example["task"]
    if task == "gsm8k":
        return _gsm8k_reward(completion, example["answer"])
    if task == "math":
        return _math_reward(completion, example["solution"])
    if task == "mbpp":
        return mbpp_pass_fail_reward(completion, example.get("test_list", []), timeout=timeout)
    raise ValueError(f"Unsupported task: {task}")


def score_completions(examples: Sequence[dict], completions: Sequence[str], *, timeout: float = 3.0) -> list[int]:
    return [score_completion(example, completion, timeout=timeout) for example, completion in zip(examples, completions)]


def pass_at_k(reward_groups: Iterable[Sequence[int]], k: int) -> Optional[float]:
    values = []
    for rewards in reward_groups:
        rewards = list(rewards)
        estimate = estimate_pass_at_k(n=len(rewards), c=sum(int(r > 0) for r in rewards), k=k)
        if estimate is not None:
            values.append(estimate)
    if not values:
        return None
    return sum(values) / len(values)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset

GSM8K_SYSTEM_PROMPT = (
    "You are a careful math solver. Work through the problem and end with a line of the form "
    "'Final answer: <number>'."
)

MATH_SYSTEM_PROMPT = (
    "Solve this math problem step by step. Put your final answer in \\boxed{}."
)

MBPP_SYSTEM_PROMPT = (
    "You are a careful Python programmer. Return only Python code inside one ```python``` block."
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_split: str
    eval_split: str
    reward_key: str


TASK_SPECS = {
    "gsm8k": TaskSpec(name="gsm8k", train_split="train", eval_split="test", reward_key="answer"),
    "math": TaskSpec(name="math", train_split="train", eval_split="test", reward_key="solution"),
    "mbpp": TaskSpec(name="mbpp", train_split="train", eval_split="test", reward_key="test_list"),
}


def _limit(ds: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def load_task_dataset(
    name: str,
    split: str,
    *,
    max_samples: Optional[int] = None,
) -> Dataset:
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)

        def _map(row):
            return {
                "task": "gsm8k",
                "prompt": (
                    f"{GSM8K_SYSTEM_PROMPT}\n\n"
                    f"Question: {row['question']}\n"
                    "Answer with reasoning, then finish with 'Final answer: <number>'."
                ),
                "answer": row["answer"],
                "question": row["question"],
            }

        return _limit(ds.map(_map), max_samples)

    if name == "math":
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)

        def _map(row):
            return {
                "task": "math",
                "prompt": (
                    f"{MATH_SYSTEM_PROMPT}\n\n"
                    f"Problem: {row['problem']}\n"
                ),
                "solution": row["solution"],
                "problem": row["problem"],
            }

        return _limit(ds.map(_map), max_samples)

    if name == "mbpp":
        ds = load_dataset("mbpp", split=split)

        def _map(row):
            tests = row.get("test_list") or []
            return {
                "task": "mbpp",
                "prompt": (
                    f"{MBPP_SYSTEM_PROMPT}\n\n"
                    f"Task: {row['text']}\n\n"
                    "Your code must satisfy these tests:\n"
                    + "\n".join(tests)
                ),
                "text": row["text"],
                "code": row.get("code", ""),
                "test_list": tests,
            }

        return _limit(ds.map(_map), max_samples)

    raise ValueError(f"Unsupported task: {name}")

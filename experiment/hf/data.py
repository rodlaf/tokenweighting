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

POLARIS_SYSTEM_PROMPT = (
    "Solve this math problem step by step. Put your final answer in \\boxed{}."
)

# Deterministic seed used ONLY to define the POLARIS train/test split. Kept
# independent of the experiment's training seed so the eval set is identical
# across all runs/seeds.
POLARIS_SPLIT_SEED = 20260427
POLARIS_EVAL_SIZE = 1000


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
    "polaris": TaskSpec(name="polaris", train_split="train", eval_split="test", reward_key="answer"),
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

    if name == "polaris":
        # POLARIS-53k ships as a single jsonl with no train/test split. We
        # build a deterministic split here so all runs share the same eval set.
        ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        ds = ds.shuffle(seed=POLARIS_SPLIT_SEED)
        eval_size = min(POLARIS_EVAL_SIZE, len(ds) // 4)
        if split in {"train", "training"}:
            ds = ds.select(range(0, len(ds) - eval_size))
        elif split in {"test", "eval", "validation"}:
            ds = ds.select(range(len(ds) - eval_size, len(ds)))
        else:
            raise ValueError(f"Unsupported polaris split: {split}")

        def _map(row):
            return {
                "task": "polaris",
                "prompt": (
                    f"{POLARIS_SYSTEM_PROMPT}\n\n"
                    f"Problem: {row['problem']}\n"
                ),
                "answer": str(row["answer"]),
                "problem": row["problem"],
                "difficulty": row.get("difficulty"),
            }

        return _limit(ds.map(_map), max_samples)

    raise ValueError(f"Unsupported task: {name}")

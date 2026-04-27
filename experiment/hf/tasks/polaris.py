from __future__ import annotations

import re
from typing import Optional

from tasks.math_task import _extract_boxed, _normalize_math_answer


_DIFFICULTY_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_polaris_difficulty(value: str | float | int | None) -> Optional[float]:
    """POLARIS difficulty is the pass rate of R1-distill-Qwen-7B, encoded as
    a string like '7/8' (7 successes out of 8). Higher = easier."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _DIFFICULTY_RE.match(str(value))
    if not match:
        return None
    num, den = int(match.group(1)), int(match.group(2))
    if den == 0:
        return None
    return num / den


def normalize_polaris_gold(answer: str) -> Optional[str]:
    """POLARIS gold answers are bare strings (no \\boxed{} wrapping). We apply
    the same normalization used for MATH so the comparison is symmetric with
    extracted predictions."""
    if answer is None:
        return None
    return _normalize_math_answer(str(answer))


def extract_polaris_prediction(completion: str) -> Optional[str]:
    """Predictions follow the MATH convention: extract content from the last
    \\boxed{...} in the completion."""
    boxed = _extract_boxed(completion)
    if boxed is None:
        return None
    return _normalize_math_answer(boxed)


def polaris_reward(prediction: str, gold_answer: str) -> int:
    pred = extract_polaris_prediction(prediction)
    gold = normalize_polaris_gold(gold_answer)
    return int(pred is not None and gold is not None and pred == gold)

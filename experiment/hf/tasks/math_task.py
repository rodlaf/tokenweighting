from __future__ import annotations

import re
from typing import Optional


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} with balanced-brace matching."""
    needle = "\\boxed{"
    last_start = text.rfind(needle)
    if last_start == -1:
        return None
    inner_start = last_start + len(needle)
    depth = 1
    i = inner_start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[inner_start : i - 1]


def _normalize_math_answer(text: str) -> str:
    """Light normalization for MATH answers: strip whitespace, collapse spaces,
    remove trailing periods, lowercase."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    text = text.replace("\\text", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.strip()
    return text


def extract_math_gold_answer(solution: str) -> Optional[str]:
    """Extract gold answer from a MATH dataset solution string."""
    answer = _extract_boxed(solution)
    if answer is None:
        return None
    return _normalize_math_answer(answer)


def extract_math_prediction(completion: str) -> Optional[str]:
    """Extract predicted answer from a model completion."""
    answer = _extract_boxed(completion)
    if answer is None:
        return None
    return _normalize_math_answer(answer)


def math_reward(prediction: str, gold_solution: str) -> int:
    pred = extract_math_prediction(prediction)
    gold = extract_math_gold_answer(gold_solution)
    return int(pred is not None and gold is not None and pred == gold)

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer|therefore|thus|so)\s*(?:is|=|:)?\s*([-+]?[$]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?[$]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_GOLD_RE = re.compile(r"####\s*([^\n]+)")


def _strip_formatting(text: str) -> str:
    text = text.strip()
    text = text.replace("\u2212", "-")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    return text.strip().rstrip(". ")


def normalize_gsm8k_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    text = _strip_formatting(str(text))
    if not text:
        return None

    if re.fullmatch(r"[-+]?\d+/\d+", text):
        try:
            frac = Fraction(text)
            return _decimal_to_string(Decimal(frac.numerator) / Decimal(frac.denominator))
        except (ZeroDivisionError, InvalidOperation):
            return text

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            return _decimal_to_string(Decimal(text))
        except InvalidOperation:
            return text

    return text.lower()


def _decimal_to_string(value: Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def extract_gsm8k_gold_answer(gold_answer: str) -> Optional[str]:
    if gold_answer is None:
        return None
    match = _GOLD_RE.search(gold_answer)
    candidate = match.group(1) if match else gold_answer
    number_matches = _NUMBER_RE.findall(candidate)
    if number_matches:
        return normalize_gsm8k_answer(number_matches[-1])
    return normalize_gsm8k_answer(candidate)


def extract_gsm8k_final_answer(prediction: str) -> Optional[str]:
    if prediction is None:
        return None
    text = prediction.strip()
    if not text:
        return None

    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return normalize_gsm8k_answer(boxed_matches[-1])

    final_matches = _FINAL_ANSWER_RE.findall(text)
    if final_matches:
        return normalize_gsm8k_answer(final_matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        number_matches = _NUMBER_RE.findall(line)
        if number_matches:
            return normalize_gsm8k_answer(number_matches[-1])

    number_matches = _NUMBER_RE.findall(text)
    if number_matches:
        return normalize_gsm8k_answer(number_matches[-1])

    return normalize_gsm8k_answer(text)


def gsm8k_reward(prediction: str, gold_answer: str) -> int:
    pred = extract_gsm8k_final_answer(prediction)
    gold = extract_gsm8k_gold_answer(gold_answer)
    return int(pred is not None and gold is not None and pred == gold)

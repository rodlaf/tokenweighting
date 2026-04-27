"""Unit tests for task-specific parsing and reward functions."""
from __future__ import annotations

import pytest

from tasks.gsm8k import (
    extract_gsm8k_final_answer,
    extract_gsm8k_gold_answer,
    gsm8k_reward,
    normalize_gsm8k_answer,
)
from tasks.math_task import (
    extract_math_gold_answer,
    extract_math_prediction,
    math_reward,
)
from tasks.mbpp import extract_python_code


class TestGsm8kNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("42", "42"),
            ("42.00", "42"),
            ("42.50", "42.5"),
            ("$1,234", "1234"),
            ("-7", "-7"),
            ("1/2", "0.5"),
            ("Paris", "paris"),
        ],
    )
    def test_values(self, raw, expected) -> None:
        assert normalize_gsm8k_answer(raw) == expected

    def test_none_input(self) -> None:
        assert normalize_gsm8k_answer(None) is None

    def test_empty_string(self) -> None:
        assert normalize_gsm8k_answer("") is None


class TestGsm8kGoldExtraction:
    def test_standard_format(self) -> None:
        assert extract_gsm8k_gold_answer("Some reasoning.\n#### 42") == "42"

    def test_comma_separated(self) -> None:
        assert extract_gsm8k_gold_answer("#### 1,234") == "1234"

    def test_falls_back_to_last_number(self) -> None:
        assert extract_gsm8k_gold_answer("The answer is 9 or maybe 10") == "10"


class TestGsm8kPredictionExtraction:
    def test_boxed(self) -> None:
        assert extract_gsm8k_final_answer(r"Working through\dots \boxed{42}") == "42"

    def test_final_answer_phrase(self) -> None:
        assert extract_gsm8k_final_answer("So the final answer is 42.") == "42"

    def test_last_line_number(self) -> None:
        assert extract_gsm8k_final_answer("Step 1: 10\nStep 2: 20\n42") == "42"

    def test_none_on_empty(self) -> None:
        assert extract_gsm8k_final_answer("") is None
        assert extract_gsm8k_final_answer(None) is None


class TestGsm8kReward:
    def test_correct(self) -> None:
        assert gsm8k_reward("Final answer: 42", "Some work\n#### 42") == 1

    def test_incorrect(self) -> None:
        assert gsm8k_reward("Final answer: 41", "#### 42") == 0

    def test_numeric_equivalence(self) -> None:
        """1/2 should match 0.5 under normalisation."""
        assert gsm8k_reward("Final answer: 0.5", "#### 1/2") == 1

    def test_empty_inputs(self) -> None:
        assert gsm8k_reward("", "#### 42") == 0


class TestMathExtraction:
    def test_basic_boxed(self) -> None:
        assert extract_math_prediction(r"So we get \boxed{42}.") == "42"

    def test_nested_braces(self) -> None:
        assert extract_math_prediction(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_picks_last_boxed(self) -> None:
        assert extract_math_prediction(r"\boxed{1}. Wait actually \boxed{2}.") == "2"

    def test_unclosed_brace_returns_none(self) -> None:
        assert extract_math_prediction(r"\boxed{unclosed") is None

    def test_no_boxed_returns_none(self) -> None:
        assert extract_math_prediction("just text with a number 5") is None

    def test_gold_extraction(self) -> None:
        assert extract_math_gold_answer(r"Working\dots \boxed{\dfrac{3}{4}}") == r"\frac{3}{4}"


class TestMathReward:
    def test_correct(self) -> None:
        assert math_reward(r"\boxed{7}", r"\boxed{7}") == 1

    def test_incorrect(self) -> None:
        assert math_reward(r"\boxed{7}", r"\boxed{8}") == 0

    def test_no_boxed_in_prediction(self) -> None:
        assert math_reward("the answer is 7", r"\boxed{7}") == 0

    def test_dfrac_equals_frac(self) -> None:
        """The normaliser should treat \\dfrac and \\frac as equivalent."""
        assert math_reward(r"\boxed{\frac{1}{2}}", r"\boxed{\dfrac{1}{2}}") == 1


class TestMbppExtraction:
    def test_python_code_block(self) -> None:
        text = "Here is the code:\n```python\ndef f(): return 1\n```"
        assert extract_python_code(text) == "def f(): return 1"

    def test_plain_code_block(self) -> None:
        text = "```\ndef f():\n    return 1\n```"
        assert "def f" in extract_python_code(text)

    def test_no_code_block_returns_raw(self) -> None:
        text = "def f(): return 1"
        assert "def f" in extract_python_code(text)

    def test_picks_last_block(self) -> None:
        text = "```python\na=1\n```\n```python\nb=2\n```"
        out = extract_python_code(text)
        assert "b=2" in out and "a=1" not in out

    def test_dedents(self) -> None:
        text = "```python\n    def f():\n        return 1\n```"
        out = extract_python_code(text)
        assert out.startswith("def f")

    def test_handles_none(self) -> None:
        assert extract_python_code(None) == ""

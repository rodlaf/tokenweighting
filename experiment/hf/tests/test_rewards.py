"""Unit tests for rewards.py and pass@k accounting."""
from __future__ import annotations

import pytest

from rewards import pass_at_k, score_completion, score_completions
from tasks.mbpp import estimate_pass_at_k


class TestEstimatePassAtK:
    def test_no_correct_samples(self) -> None:
        assert estimate_pass_at_k(n=10, c=0, k=4) == 0.0

    def test_all_correct_samples(self) -> None:
        assert estimate_pass_at_k(n=10, c=10, k=4) == pytest.approx(1.0)

    def test_k_larger_than_n(self) -> None:
        assert estimate_pass_at_k(n=4, c=2, k=5) is None

    def test_invalid_inputs(self) -> None:
        assert estimate_pass_at_k(n=0, c=0, k=1) is None
        assert estimate_pass_at_k(n=5, c=0, k=0) is None

    def test_when_failure_pool_shorter_than_k(self) -> None:
        """If we have fewer failures than k, one correct is guaranteed among the k draws."""
        assert estimate_pass_at_k(n=5, c=4, k=2) == pytest.approx(1.0)

    def test_half_correct(self) -> None:
        est = estimate_pass_at_k(n=4, c=2, k=2)
        assert est == pytest.approx(1 - (1 / 6), abs=1e-6)


class TestPassAtK:
    def test_empty_groups_returns_none(self) -> None:
        assert pass_at_k([], k=4) is None

    def test_averages_across_groups(self) -> None:
        groups = [[1, 1, 0, 0], [0, 0, 0, 0]]
        out = pass_at_k(groups, k=1)
        assert out == pytest.approx((0.5 + 0.0) / 2, abs=1e-6)

    def test_treats_positive_rewards_as_success(self) -> None:
        groups = [[2, 0, 0, 0]]
        out = pass_at_k(groups, k=1)
        assert out == pytest.approx(0.25, abs=1e-6)

    def test_ignores_invalid_k(self) -> None:
        """When k > n for every group, pass_at_k returns None."""
        groups = [[1, 0]]
        assert pass_at_k(groups, k=5) is None


class TestScoreCompletion:
    def test_gsm8k_correct(self) -> None:
        example = {"task": "gsm8k", "answer": "Some reasoning.\n#### 42"}
        completion = "The answer is 42"
        assert score_completion(example, completion) == 1

    def test_gsm8k_wrong(self) -> None:
        example = {"task": "gsm8k", "answer": "#### 42"}
        assert score_completion(example, "The answer is 41") == 0

    def test_math_correct(self) -> None:
        example = {"task": "math", "solution": r"We compute. \boxed{7}"}
        assert score_completion(example, r"Therefore the answer is \boxed{7}.") == 1

    def test_math_wrong(self) -> None:
        example = {"task": "math", "solution": r"\boxed{7}"}
        assert score_completion(example, r"\boxed{8}") == 0

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported task"):
            score_completion({"task": "unknown"}, "whatever")


class TestScoreCompletionsBatch:
    def test_batch_scoring(self) -> None:
        examples = [
            {"task": "gsm8k", "answer": "#### 7"},
            {"task": "gsm8k", "answer": "#### 10"},
        ]
        completions = ["The answer is 7", "The answer is 11"]
        assert score_completions(examples, completions) == [1, 0]

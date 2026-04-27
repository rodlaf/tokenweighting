"""Unit tests for pure helper functions in run_experiment.py.

These tests intentionally avoid anything that would load a pretrained
model or require a GPU. We only test the small, deterministic helpers.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import yaml

from run_experiment import (
    adapter_residual_norms_from_hidden,
    build_completion_mask,
    compute_advantages,
    deep_update,
    load_config,
    pick_dtype,
    validate_config,
)


class TestDeepUpdate:
    def test_merges_nested_dicts(self) -> None:
        base = {"a": 1, "b": {"x": 1, "y": 2}}
        override = {"b": {"y": 20, "z": 3}, "c": 4}
        out = deep_update(base, override)
        assert out == {"a": 1, "b": {"x": 1, "y": 20, "z": 3}, "c": 4}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}}
        base_copy = copy.deepcopy(base)
        override = {"a": {"x": 42}}
        override_copy = copy.deepcopy(override)
        _ = deep_update(base, override)
        assert base == base_copy
        assert override == override_copy

    def test_override_scalar_replaces_dict(self) -> None:
        """If override provides a non-dict value, it fully replaces the base entry."""
        base = {"a": {"x": 1}}
        out = deep_update(base, {"a": 2})
        assert out == {"a": 2}


class TestPickDtype:
    @pytest.mark.parametrize(
        "name, expected",
        [
            (None, torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
        ],
    )
    def test_valid_names(self, name, expected) -> None:
        assert pick_dtype(name) is expected

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            pick_dtype("banana")


class TestLoadConfig:
    def test_loads_plain_yaml(self, tmp_path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"a": 1, "b": {"x": 2}}))
        out = load_config(str(cfg))
        assert out == {"a": 1, "b": {"x": 2}}

    def test_merges_base_config(self, tmp_path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.safe_dump({"a": 1, "b": {"x": 1, "y": 1}}))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.safe_dump({"base_config": "base.yaml", "b": {"y": 99}, "c": 7}))
        out = load_config(str(child))
        assert out == {"a": 1, "b": {"x": 1, "y": 99}, "c": 7}
        assert "base_config" not in out

    def test_nested_base_configs(self, tmp_path) -> None:
        grand = tmp_path / "grand.yaml"
        grand.write_text(yaml.safe_dump({"a": 1}))
        base = tmp_path / "base.yaml"
        base.write_text(yaml.safe_dump({"base_config": "grand.yaml", "b": 2}))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.safe_dump({"base_config": "base.yaml", "c": 3}))
        out = load_config(str(child))
        assert out == {"a": 1, "b": 2, "c": 3}


class TestBuildCompletionMask:
    def test_masks_pad_tokens(self) -> None:
        tokens = torch.tensor([[1, 2, 0, 0]])
        mask = build_completion_mask(tokens, pad_token_id=0, eos_token_id=None)
        assert torch.allclose(mask, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))

    def test_eos_cuts_subsequent_positions(self) -> None:
        tokens = torch.tensor([[1, 2, 9, 5, 6]])
        mask = build_completion_mask(tokens, pad_token_id=0, eos_token_id=9)
        assert torch.allclose(mask, torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]))

    def test_no_eos_keeps_all(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        mask = build_completion_mask(tokens, pad_token_id=0, eos_token_id=9)
        assert torch.allclose(mask, torch.ones(1, 4))

    def test_eos_at_last_position_no_zeros_added(self) -> None:
        tokens = torch.tensor([[1, 2, 9]])
        mask = build_completion_mask(tokens, pad_token_id=0, eos_token_id=9)
        assert torch.allclose(mask, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_no_pad_token_and_no_eos_is_all_ones(self) -> None:
        tokens = torch.tensor([[1, 2, 3]])
        mask = build_completion_mask(tokens, pad_token_id=None, eos_token_id=None)
        assert torch.allclose(mask, torch.ones(1, 3))

    def test_per_row_behaviour(self) -> None:
        tokens = torch.tensor([[1, 9, 2, 0], [1, 2, 3, 4]])
        mask = build_completion_mask(tokens, pad_token_id=0, eos_token_id=9)
        assert torch.allclose(mask[0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        assert torch.allclose(mask[1], torch.ones(4))


class TestComputeAdvantages:
    def test_rloo_row_sums_to_zero(self) -> None:
        rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        adv = compute_advantages(rewards, "rloo")
        assert math.isclose(adv.sum().item(), 0.0, abs_tol=1e-6)

    def test_rloo_formula(self) -> None:
        """For K=2, RLOO advantage reduces to rewards - reversed(rewards)."""
        rewards = torch.tensor([[3.0, 1.0]])
        adv = compute_advantages(rewards, "rloo")
        assert torch.allclose(adv, torch.tensor([[2.0, -2.0]]))

    def test_rloo_needs_min_two_generations(self) -> None:
        with pytest.raises(ValueError, match="num_generations >= 2"):
            compute_advantages(torch.tensor([[1.0]]), "rloo")

    def test_grpo_zero_mean(self) -> None:
        rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        adv = compute_advantages(rewards, "grpo")
        assert math.isclose(adv.mean().item(), 0.0, abs_tol=1e-5)

    def test_grpo_unit_variance(self) -> None:
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        adv = compute_advantages(rewards, "grpo")
        assert adv.std(unbiased=True).item() == pytest.approx(1.0, abs=1e-4)

    def test_grpo_constant_rewards_does_not_nan(self) -> None:
        """When all rewards are identical, std=0; we must avoid NaN via eps."""
        rewards = torch.full((1, 4), 0.7)
        adv = compute_advantages(rewards, "grpo")
        assert torch.isfinite(adv).all()

    def test_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compute_advantages(torch.ones(1, 2), "reinforce")


class TestAdapterResidualNormsFromHidden:
    def test_zero_residual_norm(self) -> None:
        h = torch.randn(1, 5, 8)
        out = adapter_residual_norms_from_hidden(h, h)
        assert torch.allclose(out, torch.zeros(1, 5), atol=1e-6)

    def test_matches_l2_norm(self) -> None:
        adapted = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        base = torch.zeros(1, 2, 2)
        out = adapter_residual_norms_from_hidden(adapted, base)
        assert out[0, 0].item() == pytest.approx(5.0, abs=1e-5)
        assert out[0, 1].item() == pytest.approx(0.0, abs=1e-5)

    def test_output_shape(self) -> None:
        adapted = torch.randn(2, 4, 16)
        base = torch.randn(2, 4, 16)
        out = adapter_residual_norms_from_hidden(adapted, base)
        assert out.shape == (2, 4)

    def test_promotes_to_float(self) -> None:
        adapted = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        base = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        out = adapter_residual_norms_from_hidden(adapted, base)
        assert out.dtype == torch.float32


def _good_config(**overrides):
    cfg = {
        "seed": 0,
        "output_dir": "/tmp/out",
        "model": {"name_or_path": "some-model"},
        "dataset": {"name": "gsm8k", "train_split": "train", "eval_split": "test"},
        "training": {
            "algorithm": "rloo",
            "weighting": "uniform",
            "num_generations": 2,
            "batch_size": 1,
            "train_steps": 1,
            "max_prompt_length": 16,
            "max_new_tokens": 16,
            "learning_rate": 1e-4,
        },
        "evaluation": {"num_examples": 1, "pass_k": 1},
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


class TestValidateConfig:
    def test_valid_config_passes(self) -> None:
        validate_config(_good_config())

    def test_missing_top_level_key(self) -> None:
        cfg = _good_config()
        del cfg["evaluation"]
        with pytest.raises(ValueError, match="Missing top-level"):
            validate_config(cfg)

    def test_unknown_dataset(self) -> None:
        cfg = _good_config()
        cfg["dataset"]["name"] = "bogus"
        with pytest.raises(ValueError, match="Unsupported dataset"):
            validate_config(cfg)

    def test_bad_algorithm(self) -> None:
        cfg = _good_config()
        cfg["training"]["algorithm"] = "reinforce"
        with pytest.raises(ValueError, match="rloo.*grpo"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "mode",
        ["uniform", "surprisal", "entropy_reduction", "divergence", "adapter_residual"],
    )
    def test_accepts_all_supported_weighting_modes(self, mode) -> None:
        cfg = _good_config()
        cfg["training"]["weighting"] = mode
        validate_config(cfg)

    def test_rejects_removed_critic_weighting(self) -> None:
        """Critic weighting was intentionally removed; the validator must reject it."""
        cfg = _good_config()
        cfg["training"]["weighting"] = "critic"
        with pytest.raises(ValueError, match="uniform, surprisal, entropy_reduction"):
            validate_config(cfg)

    def test_num_generations_min_two(self) -> None:
        cfg = _good_config()
        cfg["training"]["num_generations"] = 1
        with pytest.raises(ValueError, match="num_generations"):
            validate_config(cfg)

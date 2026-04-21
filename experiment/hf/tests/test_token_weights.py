"""Unit tests for token_weights.py."""
from __future__ import annotations

import math

import pytest
import torch

from token_weights import (
    TokenWeightingConfig,
    _raw_scores,
    build_token_weights,
    entropy_from_logits,
    entropy_reduction_weights,
    masked_normalize,
    surprisal_weights,
    uniform_weights,
    weight_concentration_metrics,
)


def _assert_rows_sum_to_one(weights: torch.Tensor, mask: torch.Tensor) -> None:
    masked = weights * mask
    row_sums = masked.sum(dim=-1)
    has_any_mask = mask.sum(dim=-1) > 0
    for i, has in enumerate(has_any_mask.tolist()):
        if has:
            assert math.isclose(float(row_sums[i].item()), 1.0, abs_tol=1e-5)


class TestMaskedNormalize:
    def test_normalizes_rows_to_unity(self) -> None:
        w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 0.0]])
        m = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        out = masked_normalize(w, m)
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert torch.allclose(out[0], torch.tensor([1.0, 2.0, 3.0]) / 6.0)

    def test_respects_mask(self) -> None:
        w = torch.tensor([[1.0, 2.0, 3.0]])
        m = torch.tensor([[1.0, 1.0, 0.0]])
        out = masked_normalize(w, m)
        assert out[0, 2].item() == 0.0
        assert math.isclose(out[0, :2].sum().item(), 1.0, abs_tol=1e-6)

    def test_negative_values_clamped(self) -> None:
        w = torch.tensor([[-1.0, 2.0, 3.0]])
        m = torch.tensor([[1.0, 1.0, 1.0]])
        out = masked_normalize(w, m)
        assert out[0, 0].item() == 0.0
        assert math.isclose(out[0, 1:].sum().item(), 1.0, abs_tol=1e-6)

    def test_all_zero_row_does_not_nan(self) -> None:
        w = torch.zeros(1, 4)
        m = torch.ones(1, 4)
        out = masked_normalize(w, m)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out))


class TestUniformWeights:
    def test_unmasked_is_one_over_length(self) -> None:
        m = torch.ones(2, 4)
        out = uniform_weights(m)
        assert torch.allclose(out, torch.full_like(out, 0.25), atol=1e-6)

    def test_partially_masked(self) -> None:
        m = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        out = uniform_weights(m)
        expected = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)


class TestSurprisalWeights:
    def test_monotone_in_surprisal(self) -> None:
        """Tokens with larger surprisal (more negative logp) get larger weights."""
        logps = torch.tensor([[-0.1, -1.0, -5.0]])
        m = torch.ones(1, 3)
        out = surprisal_weights(logps, m)
        assert out[0, 2] > out[0, 1] > out[0, 0]

    def test_normalizes(self) -> None:
        logps = torch.tensor([[-0.1, -1.0, -5.0], [-2.0, -2.0, -2.0]])
        m = torch.ones(2, 3)
        out = surprisal_weights(logps, m)
        _assert_rows_sum_to_one(out, m)


class TestEntropyReductionWeights:
    def test_last_position_is_zero(self) -> None:
        ent = torch.tensor([[2.0, 1.0, 0.5]])
        m = torch.ones(1, 3)
        out = entropy_reduction_weights(ent, m)
        assert out[0, -1].item() == pytest.approx(0.0, abs=1e-5)

    def test_only_positive_drops_counted(self) -> None:
        ent = torch.tensor([[1.0, 5.0, 0.0]])
        m = torch.ones(1, 3)
        out = entropy_reduction_weights(ent, m)
        assert out[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert out[0, 1].item() > 0.0

    def test_normalizes(self) -> None:
        ent = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
        m = torch.ones(1, 4)
        out = entropy_reduction_weights(ent, m)
        _assert_rows_sum_to_one(out, m)


class TestEntropyFromLogits:
    def test_uniform_logits_give_log_V(self) -> None:
        V = 8
        logits = torch.zeros(1, 3, V)
        H = entropy_from_logits(logits)
        assert torch.allclose(H, torch.full_like(H, math.log(V)), atol=1e-5)

    def test_deterministic_logits_give_zero(self) -> None:
        V = 4
        logits = torch.full((1, 2, V), -1e4)
        logits[..., 0] = 1e4
        H = entropy_from_logits(logits)
        assert torch.allclose(H, torch.zeros_like(H), atol=1e-3)


class TestRawScores:
    def _mask(self) -> torch.Tensor:
        return torch.ones(1, 4)

    def test_uniform_all_ones(self) -> None:
        out = _raw_scores("uniform", self._mask(), None, None, 1e-8)
        assert torch.allclose(out, torch.ones(1, 4))

    def test_surprisal_requires_logps(self) -> None:
        with pytest.raises(ValueError, match="per_token_logps"):
            _raw_scores("surprisal", self._mask(), None, None, 1e-8)

    def test_entropy_requires_entropies(self) -> None:
        with pytest.raises(ValueError, match="entropies"):
            _raw_scores("entropy_reduction", self._mask(), None, None, 1e-8)

    def test_divergence_requires_both(self) -> None:
        with pytest.raises(ValueError, match="per_token_logps"):
            _raw_scores("divergence", self._mask(), None, None, 1e-8)
        with pytest.raises(ValueError, match="ref_token_logps"):
            _raw_scores(
                "divergence",
                self._mask(),
                torch.zeros(1, 4),
                None,
                1e-8,
            )

    def test_adapter_residual_requires_norms(self) -> None:
        with pytest.raises(ValueError, match="adapter_residual_norms"):
            _raw_scores("adapter_residual", self._mask(), None, None, 1e-8)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown weighting mode"):
            _raw_scores("nonexistent", self._mask(), None, None, 1e-8)

    def test_divergence_is_abs_log_ratio(self) -> None:
        logp = torch.tensor([[-1.0, -2.0]])
        ref = torch.tensor([[-2.0, -1.0]])
        out = _raw_scores("divergence", torch.ones(1, 2), logp, None, 0.0, ref_token_logps=ref)
        assert torch.allclose(out, torch.tensor([[1.0, 1.0]]))

    def test_adapter_residual_passes_through(self) -> None:
        norms = torch.tensor([[0.5, 1.5, 2.5]])
        out = _raw_scores("adapter_residual", torch.ones(1, 3), None, None, 0.0, adapter_residual_norms=norms)
        assert torch.allclose(out, norms)


class TestBuildTokenWeights:
    def test_uniform_normalizes(self) -> None:
        m = torch.ones(2, 5)
        cfg = TokenWeightingConfig(mode="uniform")
        out = build_token_weights(cfg, m)
        _assert_rows_sum_to_one(out, m)
        assert torch.allclose(out, torch.full_like(out, 0.2), atol=1e-6)

    def test_surprisal_normalizes(self) -> None:
        m = torch.ones(1, 3)
        cfg = TokenWeightingConfig(mode="surprisal")
        logps = torch.tensor([[-0.5, -1.0, -2.0]])
        out = build_token_weights(cfg, m, per_token_logps=logps)
        _assert_rows_sum_to_one(out, m)

    def test_sharpening_concentrates_mass(self) -> None:
        """Higher sharpening should produce more concentrated weights."""
        m = torch.ones(1, 4)
        logps = torch.tensor([[-0.1, -0.5, -1.0, -4.0]])
        flat = build_token_weights(TokenWeightingConfig(mode="surprisal", sharpening=1.0), m, per_token_logps=logps)
        sharp = build_token_weights(TokenWeightingConfig(mode="surprisal", sharpening=4.0), m, per_token_logps=logps)
        assert sharp.max().item() > flat.max().item()
        assert (sharp * sharp).sum().item() > (flat * flat).sum().item()

    def test_uniform_ignores_sharpening(self) -> None:
        """Sharpening should have no effect on uniform mode (constant vector)."""
        m = torch.ones(1, 4)
        s1 = build_token_weights(TokenWeightingConfig(mode="uniform", sharpening=1.0), m)
        s4 = build_token_weights(TokenWeightingConfig(mode="uniform", sharpening=4.0), m)
        assert torch.allclose(s1, s4)

    def test_detached_by_default(self) -> None:
        m = torch.ones(1, 3)
        logps = torch.tensor([[-0.5, -1.0, -2.0]], requires_grad=True)
        out = build_token_weights(TokenWeightingConfig(mode="surprisal"), m, per_token_logps=logps)
        assert not out.requires_grad

    def test_divergence_mode(self) -> None:
        m = torch.ones(1, 3)
        logp = torch.tensor([[-1.0, -2.0, -3.0]])
        ref = torch.tensor([[-1.0, -1.0, -1.0]])
        out = build_token_weights(
            TokenWeightingConfig(mode="divergence"),
            m,
            per_token_logps=logp,
            ref_token_logps=ref,
        )
        _assert_rows_sum_to_one(out, m)
        assert out[0, 2] > out[0, 1] > out[0, 0]

    def test_adapter_residual_mode(self) -> None:
        m = torch.ones(1, 3)
        norms = torch.tensor([[0.1, 1.0, 4.0]])
        out = build_token_weights(
            TokenWeightingConfig(mode="adapter_residual"),
            m,
            adapter_residual_norms=norms,
        )
        _assert_rows_sum_to_one(out, m)
        assert out[0, 2] > out[0, 1] > out[0, 0]


class TestWeightConcentrationMetrics:
    def test_uniform_has_low_gini_and_full_effn(self) -> None:
        m = torch.ones(2, 5)
        w = uniform_weights(m)
        metrics = weight_concentration_metrics(w, m)
        assert metrics["weight_gini"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["weight_effective_n_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_concentrated_weights_have_high_gini(self) -> None:
        m = torch.ones(1, 10)
        w = torch.zeros(1, 10)
        w[0, 0] = 1.0
        metrics = weight_concentration_metrics(w, m)
        assert metrics["weight_gini"] > 0.8
        assert metrics["weight_effective_n_ratio"] == pytest.approx(0.1, abs=1e-5)

    def test_mixed_concentration(self) -> None:
        """A partially concentrated distribution lies between uniform and one-hot."""
        m = torch.ones(1, 4)
        uniform = uniform_weights(m)
        concentrated = torch.zeros(1, 4)
        concentrated[0, 0] = 1.0
        middle = torch.tensor([[0.5, 0.25, 0.125, 0.125]])
        u = weight_concentration_metrics(uniform, m)
        c = weight_concentration_metrics(concentrated, m)
        mid = weight_concentration_metrics(middle, m)
        assert u["weight_gini"] < mid["weight_gini"] < c["weight_gini"]
        assert u["weight_effective_n_ratio"] > mid["weight_effective_n_ratio"] > c["weight_effective_n_ratio"]

    def test_ignores_masked_positions(self) -> None:
        """Masked positions should not contribute to concentration."""
        w_long = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        m_long = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        w_short = torch.tensor([[0.5, 0.5]])
        m_short = torch.tensor([[1.0, 1.0]])
        out_long = weight_concentration_metrics(w_long, m_long)
        out_short = weight_concentration_metrics(w_short, m_short)
        assert out_long["weight_gini"] == pytest.approx(out_short["weight_gini"], abs=1e-5)
        assert out_long["weight_effective_n_ratio"] == pytest.approx(
            out_short["weight_effective_n_ratio"], abs=1e-5
        )

    def test_returns_floats_not_tensors(self) -> None:
        m = torch.ones(1, 3)
        w = uniform_weights(m)
        metrics = weight_concentration_metrics(w, m)
        assert isinstance(metrics["weight_gini"], float)
        assert isinstance(metrics["weight_effective_n_ratio"], float)

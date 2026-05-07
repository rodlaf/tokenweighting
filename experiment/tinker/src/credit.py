"""
Token-level credit assignment framework.

Implements the generalized credit function:

    c_t(x, y) = alpha_t(x, y) * A(x, y) + psi_t(x, y, A)

where alpha_t is a multiplicative redistribution of the trajectory-level
advantage A, and psi_t is an additive per-token process signal (which may
itself depend on A for methods like REPO).

Methods in the literature map to:

    Uniform GRPO/RLOO:    alpha=uniform,            psi=none
    ITW-Surprisal:        alpha=surprisal,           psi=none
    ITW-EntropyReduction: alpha=entropy_reduction,   psi=none
    80/20 masking:        alpha=entropy_topk,         psi=none
    REPO-R (Petrenko+):   alpha=uniform,             psi=repo_rescale
    ERPO (Yu+):           alpha=erpo_gating,          psi=erpo_progress
    (novel combinations): alpha=surprisal,            psi=repo_rescale
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch


# ---------------------------------------------------------------------------
# Input signals
# ---------------------------------------------------------------------------

@dataclass
class TokenSignals:
    """Per-token signals available after a rollout or teacher-forced pass."""

    logprobs: torch.Tensor
    """(T,) log pi(y_t | y_{<t}, x) for each completion token."""

    mask: torch.Tensor
    """(T,) binary mask over completion tokens (0 for prompt/pad)."""

    entropies: torch.Tensor | None = None
    """(T,) per-position entropy H_t, if available from top-k."""

    ref_logprobs: torch.Tensor | None = None
    """(T,) log pi_ref(y_t | y_{<t}, x), for ERPO progress signal."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class MultiplicativeWeight(Protocol):
    """Computes alpha_t: normalized redistribution weights over tokens."""

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        """Return (T,) weights summing to 1 over masked positions."""
        ...


class AdditiveSignal(Protocol):
    """Computes psi_t: per-token credit, optionally dependent on advantage."""

    def __call__(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        """Return (T,) additive per-token credits."""
        ...


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _masked_normalize(raw: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize non-negative raw scores to sum to 1 over masked positions."""
    raw = raw.clamp(min=0.0) * mask
    return raw / (raw.sum() + eps)


def _masked_zscore(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalize over masked positions."""
    active = x[mask > 0]
    if active.numel() == 0:
        return torch.zeros_like(x)
    return ((x - active.mean()) / (active.std() + eps)) * mask


# ---------------------------------------------------------------------------
# Multiplicative weights (alpha_t)
# ---------------------------------------------------------------------------

class Uniform(MultiplicativeWeight):
    """alpha_t = 1/T. Recovers standard GRPO/RLOO."""

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        return _masked_normalize(torch.ones_like(signals.mask), signals.mask)


class Surprisal(MultiplicativeWeight):
    """alpha_t proportional to -log pi(y_t). Emphasizes low-probability tokens."""

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        raw = (-signals.logprobs).clamp(min=0.0)
        return _masked_normalize(raw, signals.mask)


class EntropyReduction(MultiplicativeWeight):
    """alpha_t proportional to max(0, H_t - H_{t+1}). Emphasizes commitment points."""

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        if signals.entropies is None:
            raise ValueError("EntropyReduction requires per-token entropies")
        H = signals.entropies
        drops = torch.zeros_like(H)
        drops[:-1] = (H[:-1] - H[1:]).clamp(min=0.0)
        return _masked_normalize(drops, signals.mask)


class EntropyMagnitude(MultiplicativeWeight):
    """alpha_t proportional to H_t. Emphasizes high-uncertainty positions."""

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        if signals.entropies is None:
            raise ValueError("EntropyMagnitude requires per-token entropies")
        return _masked_normalize(signals.entropies, signals.mask)


class TopKMask(MultiplicativeWeight):
    """alpha_t = 1 for top-p% highest entropy positions, 0 elsewhere.

    Approximates the hard masking in Wang et al. (Beyond 80/20).
    """

    def __init__(self, percentile: float = 0.8):
        self.percentile = percentile

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        if signals.entropies is None:
            raise ValueError("TopKMask requires per-token entropies")
        H = signals.entropies * signals.mask
        active = signals.mask.sum().long().item()
        k = max(1, int(active * (1 - self.percentile)))
        threshold = H[signals.mask > 0].topk(k).values[-1]
        raw = (H >= threshold).float()
        return _masked_normalize(raw, signals.mask)


class ERPOGating(MultiplicativeWeight):
    """alpha_t = sigma(gamma * z-score(H_t)).

    ERPO's entropy-aware gating (Eq 8, Yu et al. 2603.28204).
    Requires group-level entropy stats to be pre-computed into z-scores
    and passed via entropies field, or computes per-trajectory z-score
    as a fallback.
    """

    def __init__(self, gamma: float = 5.0):
        self.gamma = gamma

    def __call__(self, signals: TokenSignals) -> torch.Tensor:
        if signals.entropies is None:
            raise ValueError("ERPOGating requires per-token entropies")
        z = _masked_zscore(signals.entropies, signals.mask)
        gated = torch.sigmoid(self.gamma * z)
        return _masked_normalize(gated, signals.mask)


# ---------------------------------------------------------------------------
# Additive signals (psi_t)
# ---------------------------------------------------------------------------

class NoAdditive(AdditiveSignal):
    """psi_t = 0. No additive process signal."""

    def __call__(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        return torch.zeros_like(signals.mask)


class CenteredLogprob(AdditiveSignal):
    """psi_t = -beta * (log pi(y_t) - mean log pi).

    Simplified REPO-style correction with trajectory-level centering.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def __call__(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        lp = signals.logprobs * signals.mask
        mean_lp = lp.sum() / signals.mask.sum().clamp(min=1)
        centered = lp - mean_lp
        return -self.beta * centered * signals.mask


class REPORescale(AdditiveSignal):
    """REPO-R (Petrenko et al. 2603.11682): psi_t = -zeta * |A| * L_t.

    L_t = log pi(y_t) - E[log pi(.|y_{<t})] = log pi(y_t) + H_t

    This is the mean-centered log-probability where centering is across
    the vocabulary (not the trajectory). Requires per-token entropies.

    The adaptive controller that adjusts zeta to maintain initial entropy
    lives in the training loop, not here.
    """

    def __init__(self, zeta: float = 1.0):
        self.zeta = zeta

    def __call__(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        if signals.entropies is None:
            raise ValueError("REPORescale requires per-token entropies for vocabulary-level centering")
        # L_t = log pi(y_t) + H_t  (mean-centered across vocab)
        L = (signals.logprobs + signals.entropies) * signals.mask
        return -self.zeta * abs(advantage) * L * signals.mask


class ERPOProgress(AdditiveSignal):
    """ERPO's result-anchored progress signal (Eq 7+10, Yu et al. 2603.28204).

    psi_t = eta * W_t * sgn(A) * s_t

    where s_t = beta_progress * (log pi_theta - log pi_ref) is the implicit
    progress signal and W_t is entropy-aware gating. When used with
    ERPOGating as alpha, the gating is already in the multiplicative weights,
    so this class only handles the progress signal with outcome anchoring.

    Bucket normalization of s_t should be done in the training loop before
    passing to this class (set ref_logprobs to the already-normalized values).
    """

    def __init__(self, eta: float = 0.2, beta_progress: float = 0.1):
        self.eta = eta
        self.beta_progress = beta_progress

    def __call__(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        if signals.ref_logprobs is None:
            raise ValueError("ERPOProgress requires ref_logprobs")
        # Progress signal: how much has the policy moved from reference?
        progress = self.beta_progress * (signals.logprobs - signals.ref_logprobs) * signals.mask
        # Outcome anchoring: only push in the direction the outcome warrants
        sign_A = 1.0 if advantage >= 0 else -1.0
        return self.eta * sign_A * progress


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@dataclass
class CreditFunction:
    """Composes multiplicative and additive components into per-token credits.

        c_t = alpha_t * A * T + psi_t(signals, A)

    where T = number of active tokens, so that the total gradient magnitude
    matches the uniform baseline.
    """

    alpha: MultiplicativeWeight = field(default_factory=Uniform)
    psi: AdditiveSignal = field(default_factory=NoAdditive)

    def compute(self, signals: TokenSignals, advantage: float) -> torch.Tensor:
        """Return (T,) per-token credit values c_t."""
        T = signals.mask.sum().clamp(min=1).item()
        weights = self.alpha(signals)
        additive = self.psi(signals, advantage)
        return weights * advantage * T + additive


# ---------------------------------------------------------------------------
# Adaptive entropy controller (for REPO)
# ---------------------------------------------------------------------------

class AdaptiveZetaController:
    """Tracks batch entropy and adjusts zeta to maintain initial level.

    Used with REPORescale. Call update() each training step with the
    current batch mean entropy. The controller doubles/halves zeta
    to keep entropy near the initial value.
    """

    def __init__(self, initial_zeta: float = 1.0):
        self.zeta = initial_zeta
        self.target_entropy: float | None = None

    def update(self, batch_mean_entropy: float) -> float:
        if self.target_entropy is None:
            self.target_entropy = batch_mean_entropy
            return self.zeta
        if batch_mean_entropy < self.target_entropy:
            self.zeta *= 2.0  # entropy too low, increase correction
        else:
            self.zeta *= 0.5  # entropy too high, decrease correction
        self.zeta = max(0.01, min(self.zeta, 100.0))  # clamp
        return self.zeta


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALPHAS: dict[str, type] = {
    "uniform": Uniform,
    "surprisal": Surprisal,
    "entropy_reduction": EntropyReduction,
    "entropy_magnitude": EntropyMagnitude,
    "topk_mask": TopKMask,
    "erpo_gating": ERPOGating,
}

PSIS: dict[str, type] = {
    "none": NoAdditive,
    "centered_logprob": CenteredLogprob,
    "repo_rescale": REPORescale,
    "erpo_progress": ERPOProgress,
}


def build_credit_function(
    alpha: str = "uniform",
    psi: str = "none",
    alpha_kwargs: dict | None = None,
    psi_kwargs: dict | None = None,
) -> CreditFunction:
    """Construct a CreditFunction from string names.

    >>> build_credit_function("surprisal")
    >>> build_credit_function("uniform", "repo_rescale", psi_kwargs={"zeta": 1.0})
    >>> build_credit_function("erpo_gating", "erpo_progress", psi_kwargs={"eta": 0.2})
    """
    alpha_obj = ALPHAS[alpha](**(alpha_kwargs or {}))
    psi_obj = PSIS[psi](**(psi_kwargs or {}))
    return CreditFunction(alpha=alpha_obj, psi=psi_obj)

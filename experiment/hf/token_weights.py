from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class TokenWeightingConfig:
    mode: str = "uniform"  # uniform | surprisal | entropy_reduction
    eps: float = 1e-8
    sharpening: float = 1.0  # raise raw weights to this power before normalizing
    detach: bool = True


def masked_normalize(weights: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(weights.dtype)
    weights = torch.clamp(weights, min=0.0) * mask
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return weights / denom


def uniform_weights(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return masked_normalize(torch.ones_like(mask, dtype=torch.float32), mask, eps=eps)


def surprisal_weights(per_token_logps: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    surprisal = (-per_token_logps).float()
    return masked_normalize(surprisal + eps, mask, eps=eps)


def entropy_reduction_weights(entropies: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    next_entropies = torch.zeros_like(entropies)
    next_entropies[:, :-1] = entropies[:, 1:]
    drops = torch.clamp(entropies - next_entropies, min=0.0)
    drops[:, -1] = 0.0  # Delta H_T = 0 for the final token
    return masked_normalize(drops + eps, mask, eps=eps)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _raw_scores(
    mode: str,
    completion_mask: torch.Tensor,
    per_token_logps: Optional[torch.Tensor],
    entropies: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Compute unnormalized salience scores for each token."""
    if mode == "uniform":
        return torch.ones_like(completion_mask, dtype=torch.float32)
    if mode == "surprisal":
        if per_token_logps is None:
            raise ValueError("per_token_logps required for surprisal weighting")
        return (-per_token_logps).float() + eps
    if mode == "entropy_reduction":
        if entropies is None:
            raise ValueError("entropies required for entropy_reduction weighting")
        next_ent = torch.zeros_like(entropies)
        next_ent[:, :-1] = entropies[:, 1:]
        drops = torch.clamp(entropies - next_ent, min=0.0)
        drops[:, -1] = 0.0
        return drops + eps
    raise ValueError(f"Unknown weighting mode: {mode}")


def build_token_weights(
    cfg: TokenWeightingConfig,
    completion_mask: torch.Tensor,
    *,
    per_token_logps: Optional[torch.Tensor] = None,
    entropies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scores = _raw_scores(cfg.mode, completion_mask, per_token_logps, entropies, cfg.eps)
    if cfg.sharpening != 1.0 and cfg.mode != "uniform":
        scores = scores.pow(cfg.sharpening)
    weights = masked_normalize(scores, completion_mask, eps=cfg.eps)
    return weights.detach() if cfg.detach else weights

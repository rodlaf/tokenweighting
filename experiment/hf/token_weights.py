from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class TokenWeightingConfig:
    mode: str = "uniform"  # uniform | surprisal | entropy_reduction | divergence | adapter_residual
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


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 2) -> torch.Tensor:
    """Compute per-token entropy H = -sum(p * log p) chunked along the batch dim
    to avoid materializing a full [B, T, V] fp32 softmax tensor at once."""
    if logits.dim() != 3:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)
    B = logits.size(0)
    out = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        lp = F.log_softmax(logits[start:end].float(), dim=-1)
        out[start:end] = -(lp.exp() * lp).sum(dim=-1)
        del lp
    return out


def _raw_scores(
    mode: str,
    completion_mask: torch.Tensor,
    per_token_logps: Optional[torch.Tensor],
    entropies: Optional[torch.Tensor],
    eps: float,
    ref_token_logps: Optional[torch.Tensor] = None,
    adapter_residual_norms: Optional[torch.Tensor] = None,
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
    if mode == "divergence":
        if per_token_logps is None:
            raise ValueError("per_token_logps required for divergence weighting")
        if ref_token_logps is None:
            raise ValueError("ref_token_logps required for divergence weighting")
        return (per_token_logps - ref_token_logps).abs().float() + eps
    if mode == "adapter_residual":
        if adapter_residual_norms is None:
            raise ValueError("adapter_residual_norms required for adapter_residual weighting")
        return adapter_residual_norms.float() + eps
    raise ValueError(f"Unknown weighting mode: {mode}")


def build_token_weights(
    cfg: TokenWeightingConfig,
    completion_mask: torch.Tensor,
    *,
    per_token_logps: Optional[torch.Tensor] = None,
    entropies: Optional[torch.Tensor] = None,
    ref_token_logps: Optional[torch.Tensor] = None,
    adapter_residual_norms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scores = _raw_scores(
        cfg.mode,
        completion_mask,
        per_token_logps,
        entropies,
        cfg.eps,
        ref_token_logps=ref_token_logps,
        adapter_residual_norms=adapter_residual_norms,
    )
    if cfg.sharpening != 1.0 and cfg.mode != "uniform":
        scores = scores.pow(cfg.sharpening)
    weights = masked_normalize(scores, completion_mask, eps=cfg.eps)
    return weights.detach() if cfg.detach else weights


def weight_concentration_metrics(weights: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute concentration/dispersion metrics for token weights.

    Returns:
        gini: Gini coefficient averaged across the batch (0=uniform, 1=all-on-one-token)
        effective_n_ratio: mean(1 / (L * sum(w_t^2))), where L is completion length.
            Equals 1.0 when uniform; tends to 0 when concentrated on a few tokens.
    """
    w = weights * mask
    lengths = mask.sum(dim=-1).clamp_min(1.0)

    sum_sq = (w * w).sum(dim=-1).clamp_min(1e-12)
    effective_n = 1.0 / sum_sq
    effective_n_ratio = (effective_n / lengths).mean().item()

    batch = w.size(0)
    ginis = []
    for i in range(batch):
        vals = w[i][mask[i] > 0]
        if vals.numel() <= 1:
            continue
        sorted_vals, _ = torch.sort(vals)
        n = sorted_vals.numel()
        idx = torch.arange(1, n + 1, device=vals.device, dtype=vals.dtype)
        numerator = (2 * idx - n - 1) * sorted_vals
        denom = n * sorted_vals.sum().clamp_min(1e-12)
        ginis.append((numerator.sum() / denom).item())
    gini_mean = float(sum(ginis) / len(ginis)) if ginis else 0.0

    return {
        "weight_gini": gini_mean,
        "weight_effective_n_ratio": float(effective_n_ratio),
    }

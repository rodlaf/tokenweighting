
from __future__ import annotations

import numpy as np


def compute_token_weights(mode: str, logprobs: list[float], entropies: list[float], min_weight: float = 0.05) -> list[float]:
    if mode == 'uniform':
        raw = np.ones(len(logprobs), dtype=float)
    elif mode == 'surprisal':
        raw = np.maximum(-np.array(logprobs, dtype=float), 0.0)
    elif mode == 'entropy_reduction':
        raw = np.zeros(len(entropies), dtype=float)
        for i in range(len(entropies) - 1):
            raw[i] = max(0.0, entropies[i] - entropies[i + 1])
        raw[-1] = 0.0
    else:
        raise ValueError(f'Unknown weighting mode: {mode}')
    raw = raw + float(min_weight)
    if raw.sum() <= 0:
        return [1.0 / len(raw)] * len(raw)
    weights = raw / raw.sum()
    return [float(x) for x in weights]

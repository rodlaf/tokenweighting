
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        'mean': float(arr.mean()) if arr.size else 0.0,
        'std': float(arr.std(ddof=0)) if arr.size else 0.0,
        'min': float(arr.min()) if arr.size else 0.0,
        'max': float(arr.max()) if arr.size else 0.0,
    }

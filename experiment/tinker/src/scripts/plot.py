"""
Plot reward curves across runs from train_metrics.jsonl files.

Usage (via cli.py):
    tw plot outputs/demo/uniform outputs/demo/surprisal outputs/demo/entropy-reduction
    tw plot outputs/demo/* --metric batch_mean_entropy
    tw plot outputs/demo/* --smooth 5 --save reward_curves.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


LABEL_MAP = {
    "reward_mean": "Mean Reward (fraction correct)",
    "batch_mean_entropy": "Batch Mean Entropy",
    "n_datums": "Training Datums",
    "n_degenerate": "Degenerate Groups",
    "zeta": "Adaptive \u03b6",
}

NAME_MAP = {
    "uniform": "Uniform (baseline)",
    "surprisal": "Surprisal",
    "entropy-reduction": "Entropy Reduction",
    "entropy-magnitude": "Entropy Magnitude",
    "repo-r": "REPO-R",
    "erpo": "ERPO",
    "uniform+repo": "Uniform + REPO",
    "surprisal+repo": "Surprisal + REPO",
    "entropy-reduction+repo": "Entropy Reduction + REPO",
}


def load_metrics(run_dir: str) -> tuple[str, list[dict]]:
    p = Path(run_dir)
    name = p.name
    metrics_file = p / "train_metrics.jsonl"
    if not metrics_file.exists():
        return name, []
    records = [json.loads(line) for line in metrics_file.read_text().strip().split("\n") if line]
    return name, records


def _ema_smooth(values: list[float], span: int) -> np.ndarray:
    """Exponential moving average smoothing."""
    arr = np.array(values, dtype=float)
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def plot_runs(
    run_dirs: list[str],
    metric: str = "reward_mean",
    output: str | None = None,
    smooth: int = 5,
):
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    fig, ax = plt.subplots(figsize=(9, 5))

    for run_dir in sorted(run_dirs):
        name, records = load_metrics(run_dir)
        if not records:
            print(f"  skipping {name} (no metrics)")
            continue
        steps = [r["step"] for r in records if metric in r]
        values = [r[metric] for r in records if metric in r]
        if not steps:
            print(f"  skipping {name} (no '{metric}' in metrics)")
            continue

        label = NAME_MAP.get(name, name)
        smoothed = _ema_smooth(values, span=smooth)
        ax.plot(steps, smoothed, linewidth=2, label=label)
        ax.fill_between(steps, values, smoothed, alpha=0.1)

    ax.set_xlabel("Step")
    ax.set_ylabel(LABEL_MAP.get(metric, metric))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("GSM8K Training: Reward by Credit Assignment Method")
    sns.despine()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()

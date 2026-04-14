"""
Central CLI for token-weighting experiments on Tinker.

Usage:
    tw train --config configs/gsm8k-surprisal.yaml
    tw train --alpha surprisal --steps 10
    tw train --alpha erpo_gating --psi erpo_progress --model Qwen/Qwen3-8B
    tw smoke
"""

from __future__ import annotations

import asyncio

import click
from dotenv import load_dotenv

load_dotenv()

# Keep in sync with credit.py registries
_ALPHAS = ["uniform", "surprisal", "entropy_reduction", "entropy_magnitude", "topk_mask", "erpo_gating"]
_PSIS = ["none", "centered_logprob", "repo_rescale", "erpo_progress"]


@click.group()
def main():
    """Token-weighting experiments on Tinker."""


@main.command()
@click.option("--config", type=click.Path(exists=True), help="YAML config file")
@click.option("--model", default=None, help="Model name (overrides config)")
@click.option("--algorithm", default=None, type=click.Choice(["grpo", "rloo", "dapo", "cispo"]), help="Base RL algorithm")
@click.option("--alpha", default=None, type=click.Choice(_ALPHAS), help="Multiplicative weight")
@click.option("--psi", default=None, type=click.Choice(_PSIS), help="Additive signal")
@click.option("--steps", default=None, type=int, help="Training steps")
@click.option("--lr", default=None, type=float, help="Learning rate")
@click.option("--group-size", default=None, type=int, help="Group size")
@click.option("--output", default=None, type=str, help="Output directory")
@click.option("--wandb", "wandb_project", default=None, type=str, help="W&B project name (enables logging)")
def train(config, model, algorithm, alpha, psi, steps, lr, group_size, output, wandb_project):
    """Run a training experiment with pluggable algorithm and credit assignment."""
    from src.scripts.grpo import load_config, run_training

    if config:
        cfg = load_config(config)
    else:
        cfg = {
            "model": {"name": "meta-llama/Llama-3.2-1B", "renderer": "llama3", "lora_rank": 32},
            "training": {
                "train_steps": 100,
                "batch_size": 16,
                "group_size": 8,
                "max_new_tokens": 256,
                "learning_rate": 4e-5,
                "temperature": 0.8,
                "topk_entropy": 50,
            },
            "credit": {"alpha": "uniform", "psi": "none"},
            "output_dir": "outputs/default",
        }

    if model:
        cfg["model"]["name"] = model
    if algorithm:
        cfg["training"]["algorithm"] = algorithm
    if alpha:
        cfg.setdefault("credit", {})["alpha"] = alpha
    if psi:
        cfg.setdefault("credit", {})["psi"] = psi
    if steps:
        cfg["training"]["train_steps"] = steps
    if lr:
        cfg["training"]["learning_rate"] = lr
    if group_size:
        cfg["training"]["group_size"] = group_size
    if output:
        cfg["output_dir"] = output

    if wandb_project:
        cfg["wandb_project"] = wandb_project

    # Auto-enable features implied by method choice
    if alpha in ("erpo_gating",) or psi in ("erpo_progress",):
        cfg["model"].setdefault("reference", True)
    if psi in ("repo_rescale",):
        cfg["training"].setdefault("adaptive_zeta", True)

    if not config and not output:
        a = cfg.get("credit", {}).get("alpha", "uniform")
        p = cfg.get("credit", {}).get("psi", "none")
        cfg["output_dir"] = f"outputs/{a}-{p}"

    asyncio.run(run_training(cfg))


@main.command()
@click.argument("run_dirs", nargs=-1, required=True)
@click.option("--metric", default="reward_mean", help="Metric to plot")
@click.option("--smooth", default=5, type=int, help="EMA smoothing span (1 = no smoothing)")
@click.option("--save", default=None, type=str, help="Save to file instead of showing")
def plot(run_dirs, metric, smooth, save):
    """Plot metrics across runs. Pass output directories as arguments."""
    from src.scripts.plot import plot_runs

    plot_runs(list(run_dirs), metric=metric, output=save, smooth=smooth)


@main.command()
def smoke():
    """Run end-to-end pipeline smoke test."""
    from tests.smoke_test import run

    asyncio.run(run())


if __name__ == "__main__":
    main()

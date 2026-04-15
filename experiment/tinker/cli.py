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
def train(config, model, algorithm, alpha, psi, steps, lr, group_size, output):
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
@click.option("--config", type=click.Path(exists=True), default="configs/sweep/base.yaml", help="Base config (fixed params)")
@click.option("--scientific", "scientific_params", multiple=True, help="Scientific params: 'param=val1,val2,...'")
@click.option("--nuisance", "nuisance_params", multiple=True, help="Nuisance params: 'param=val1,val2,...'")
@click.option("--fixed", "fixed_params", default="", help="Fixed param overrides: 'param1=val1 param2=val2'")
@click.option("--study-name", required=True, help="Name for this study")
@click.option("--steps", default=200, type=int, help="Steps per trial")
@click.option("--n-trials", default=None, type=int, help="Max trials (default: full grid)")
@click.option("--sampling-method", type=click.Choice(["grid", "random"]), default="grid")
@click.option("--output", default="outputs/sweep", help="Output root directory")
@click.option("--dry-run", is_flag=True, help="Print plan without launching")
def sweep(config, scientific_params, nuisance_params, fixed_params, study_name, steps, n_trials, sampling_method, output, dry_run):
    """Launch a hyperparameter sweep.

    Params are organized per the deep learning tuning playbook:

    \b
    --scientific: what you're investigating (defines the experiment)
    --nuisance:   need to tune per method, not the focus
    --fixed:      held constant, override base config
    Base config:  everything else (model, group_size, etc.)

    \b
    Examples:
        # LR sweep across credit methods (9 trials)
        tw sweep --study-name lr-sweep-v1 \\
            --scientific "alpha=uniform,surprisal,entropy_reduction" \\
            --nuisance "lr=2e-5,4e-5,8e-5"

    \b
        # Add DAPO comparison
        tw sweep --study-name algo-comparison \\
            --scientific "alpha=uniform,surprisal,entropy_reduction" \\
            --scientific "algorithm=grpo,dapo" \\
            --nuisance "lr=2e-5,4e-5,8e-5"

    \b
        # Quick test
        tw sweep --study-name test --steps 10 --dry-run \\
            --scientific "alpha=uniform,surprisal" \\
            --nuisance "lr=4e-5"
    """
    from src.scripts.sweep import (
        generate_grid,
        launch_sweep,
        parse_fixed_params,
        parse_param_spec,
        print_sweep_plan,
    )

    scientific_space = {}
    for spec in scientific_params:
        param, values = parse_param_spec(spec)
        scientific_space[param] = values

    nuisance_space = {}
    for spec in nuisance_params:
        param, values = parse_param_spec(spec)
        nuisance_space[param] = values

    fixed = parse_fixed_params(fixed_params)

    combined = {**scientific_space, **nuisance_space}
    if not combined:
        click.echo("Error: specify at least one --scientific or --nuisance param", err=True)
        raise SystemExit(1)

    trials = generate_grid(combined, n_trials, sampling_method)
    print_sweep_plan(scientific_space, nuisance_space, fixed, trials, study_name, config, steps)
    launch_sweep(
        trials, scientific_space, nuisance_space, fixed,
        base_config=config, steps=steps, study_name=study_name,
        output_root=output, dry_run=dry_run,
    )


@main.command()
def smoke():
    """Run end-to-end pipeline smoke test."""
    from tests.smoke_test import run

    asyncio.run(run())


if __name__ == "__main__":
    main()

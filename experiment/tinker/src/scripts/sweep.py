"""
Hyperparameter sweep launcher.

Organizes parameters per the deep learning tuning playbook:
- Scientific: what you're investigating
- Nuisance: need to tune per method, not the focus
- Fixed: held constant, defined in base config

All trials launch concurrently (Tinker handles the compute).
"""

from __future__ import annotations

import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click


def parse_param_spec(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise ValueError(f"Expected param=val1,val2,... format, got: {spec}")
    param, values_str = spec.split("=", 1)
    return param, [v.strip() for v in values_str.split(",") if v.strip()]


def parse_fixed_params(fixed_str: str) -> dict[str, str]:
    if not fixed_str:
        return {}
    params = {}
    for kv in fixed_str.split():
        if "=" not in kv:
            raise ValueError(f"Expected param=value format, got: {kv}")
        k, v = kv.split("=", 1)
        params[k] = v
    return params


def generate_grid(
    search_space: dict[str, list[str]],
    n_trials: int | None,
    method: str = "grid",
) -> list[dict[str, str]]:
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    all_configs = [dict(zip(param_names, values)) for values in itertools.product(*param_values)]

    if method == "random" or (n_trials and len(all_configs) > n_trials):
        click.echo(f"Grid has {len(all_configs)} points, sampling {n_trials} randomly")
        random.shuffle(all_configs)
        return all_configs[:n_trials]
    return all_configs


_PARAM_TO_FLAG = {
    "alpha": "--alpha",
    "psi": "--psi",
    "lr": "--lr",
    "learning_rate": "--lr",
    "algorithm": "--algorithm",
    "group_size": "--group-size",
    "group-size": "--group-size",
    "model": "--model",
}


def _trial_name(trial: dict[str, str]) -> str:
    return "-".join(f"{k}{v}" for k, v in sorted(trial.items()))


def print_sweep_plan(
    scientific_space: dict[str, list[str]],
    nuisance_space: dict[str, list[str]],
    fixed_params: dict[str, str],
    trials: list[dict[str, str]],
    study_name: str,
    base_config: str,
    steps: int,
) -> None:
    w = shutil.get_terminal_size().columns
    box_w = w - 2

    click.echo()
    click.echo(click.style("+" + "=" * box_w + "+", fg="cyan"))
    click.echo(click.style("|" + f" SWEEP: {study_name}".center(box_w) + "|", fg="cyan", bold=True))
    click.echo(click.style("+" + "=" * box_w + "+", fg="cyan"))
    click.echo()

    click.echo(click.style("Study:", fg="yellow", bold=True))
    click.echo(f"  Config:       {click.style(base_config, fg='blue')}")
    click.echo(f"  Steps/trial:  {click.style(str(steps), fg='green', bold=True)}")
    click.echo(f"  Total trials: {click.style(str(len(trials)), fg='green', bold=True)}")
    click.echo()

    if scientific_space:
        click.echo(click.style("Scientific:", fg="green"))
        for param, values in scientific_space.items():
            click.echo(f"  {param}: {', '.join(values)}")

    if nuisance_space:
        click.echo(click.style("Nuisance:", fg="blue"))
        for param, values in nuisance_space.items():
            click.echo(f"  {param}: {', '.join(values)}")

    if fixed_params:
        click.echo(click.style("Fixed:", fg="white"))
        for param, value in fixed_params.items():
            click.echo(f"  {param} = {value}")

    click.echo()
    click.echo("-" * w)
    for i, trial in enumerate(trials):
        name = _trial_name(trial)
        param_str = ", ".join(f"{k}={v}" for k, v in trial.items())
        click.echo(f"  [{i+1:>2}/{len(trials)}] {click.style(name, fg='cyan')}: {param_str}")
    click.echo("-" * w)
    click.echo()


def launch_sweep(
    trials: list[dict[str, str]],
    scientific_space: dict[str, list[str]],
    nuisance_space: dict[str, list[str]],
    fixed_params: dict[str, str],
    base_config: str,
    steps: int,
    study_name: str,
    output_root: str = "outputs/sweep",
    dry_run: bool = False,
    **_kwargs,
) -> None:
    output_root_path = Path(output_root) / study_name
    output_root_path.mkdir(parents=True, exist_ok=True)

    # Resolve cwd — subprocess must run from the tinker package root
    tinker_root = Path(__file__).resolve().parent.parent.parent

    manifest = {
        "study_name": study_name,
        "base_config": base_config,
        "steps": steps,
        "scientific_params": scientific_space,
        "nuisance_params": nuisance_space,
        "fixed_params": fixed_params,
        "trials": trials,
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_root_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if dry_run:
        click.echo(click.style("DRY RUN -- manifest saved, no jobs launched.", fg="yellow", bold=True))
        return

    # Fire all jobs concurrently — Tinker handles the compute
    procs: list[tuple[str, Path, subprocess.Popen]] = []

    click.echo(f"Launching {len(trials)} trials...\n")

    for i, trial in enumerate(trials):
        name = _trial_name(trial)
        output_dir = output_root_path / name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_root_path / f"{name}.log"

        cmd = [
            sys.executable, "-m", "cli", "train",
            "--config", base_config,
            "--steps", str(steps),
            "--output", str(output_dir),
        ]
        for param, value in {**trial, **fixed_params}.items():
            flag = _PARAM_TO_FLAG.get(param)
            if flag:
                cmd.extend([flag, value])

        click.echo(click.style(f"  [{i+1:>2}/{len(trials)}] {name}", fg="cyan"))
        proc = subprocess.Popen(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(tinker_root),
        )
        procs.append((name, log_path, proc))

    click.echo(f"\nAll {len(trials)} trials launched. Waiting for completion...\n")

    # Wait for all to finish and report
    completed = 0
    failed = 0
    for name, log_path, proc in procs:
        proc.wait()
        if proc.returncode == 0:
            completed += 1
            click.echo(click.style(f"  {name}: done", fg="green"))
        else:
            failed += 1
            # Surface the actual error
            click.echo(click.style(f"  {name}: FAILED (exit {proc.returncode})", fg="red"))
            try:
                log_tail = log_path.read_text().strip().split("\n")[-5:]
                for line in log_tail:
                    click.echo(click.style(f"    {line}", fg="red", dim=True))
            except Exception:
                pass

    click.echo()
    status_color = "green" if failed == 0 else "yellow"
    click.echo(click.style(
        f"Done: {completed} succeeded, {failed} failed out of {len(trials)} trials.",
        fg=status_color, bold=True,
    ))
    click.echo(f"Results: {output_root_path}")
    click.echo(f"Compare: tw plot {output_root_path}/*")

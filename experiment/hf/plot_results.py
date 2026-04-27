"""Generate figures from experiment outputs.

Reads train_metrics.jsonl and eval_step_*.json from each run directory
under outputs/, produces PNGs into figures/.

Usage:
    python plot_results.py                    # default: outputs/ -> figures/
    python plot_results.py --outputs-dir ...  # custom paths
    python plot_results.py --figures-dir ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "uniform": "#4477AA",
    "surprisal": "#EE6677",
    "entropy": "#228833",
}
ALGO_LINESTYLE = {
    "rloo": "-",
    "grpo": "--",
}


def parse_run_name(name: str) -> dict[str, str] | None:
    """Extract algorithm and weighting from a run directory name."""
    m = re.search(r"(rloo|grpo)-(uniform|surprisal|entropy)", name)
    if not m:
        return None
    return {"algorithm": m.group(1), "weighting": m.group(2)}


def load_train_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    return df


def load_eval_checkpoints(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in run_dir.glob("eval_step_*.json"):
        step = int(f.stem.split("_")[-1])
        data = json.loads(f.read_text())
        rows.append({"step": step, **data})
    final = run_dir / "final_eval.json"
    if final.exists():
        data = json.loads(final.read_text())
        last_step = max((r["step"] for r in rows), default=0)
        if not any(r["step"] == last_step for r in rows if r.get("greedy_accuracy") == data.get("greedy_accuracy")):
            rows.append({"step": last_step, "source": "final", **data})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)


def label_for(algo: str, weighting: str) -> str:
    return f"{algo.upper()} + {weighting}"


def style_for(algo: str, weighting: str) -> dict:
    return {
        "color": COLORS.get(weighting, "#999999"),
        "linestyle": ALGO_LINESTYLE.get(algo, "-"),
        "linewidth": 1.8,
        "alpha": 0.85,
    }


def discover_runs(outputs_dir: Path) -> dict[str, dict]:
    runs = {}
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run_name(d.name)
        if meta is None:
            continue
        metrics_file = d / "train_metrics.jsonl"
        if not metrics_file.exists():
            continue
        df = load_train_metrics(metrics_file)
        if len(df) < 5:
            continue
        evals = load_eval_checkpoints(d)
        runs[d.name] = {**meta, "train": df, "evals": evals}
    return runs


def plot_training_curves(runs: dict[str, dict], figures_dir: Path) -> None:
    metrics = ["reward_mean", "loss", "mean_completion_length", "mean_surprisal"]
    titles = ["Reward Mean", "Loss", "Completion Length", "Surprisal"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col, title in zip(axes.flat, metrics, titles):
        for name, run in runs.items():
            df = run["train"]
            if col not in df.columns:
                continue
            smoothed = df[col].rolling(window=10, min_periods=1).mean()
            ax.plot(df["step"], smoothed, label=label_for(run["algorithm"], run["weighting"]),
                    **style_for(run["algorithm"], run["weighting"]))
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(runs), 3),
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(figures_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  training_curves.png  ({len(runs)} runs)")


def plot_eval_over_time(runs: dict[str, dict], figures_dir: Path) -> None:
    runs_with_evals = {k: v for k, v in runs.items() if len(v["evals"]) > 0}
    if not runs_with_evals:
        print("  (no eval checkpoints found, skipping eval_over_time)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in [
        ("greedy_accuracy", axes[0], "Greedy Accuracy"),
        ("pass@4", axes[1], "Pass@4"),
    ]:
        for name, run in runs_with_evals.items():
            edf = run["evals"]
            if metric not in edf.columns:
                continue
            ax.plot(edf["step"], edf[metric], marker="o", markersize=4,
                    label=label_for(run["algorithm"], run["weighting"]),
                    **style_for(run["algorithm"], run["weighting"]))
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(runs_with_evals), 3),
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(figures_dir / "eval_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  eval_over_time.png   ({len(runs_with_evals)} runs)")


def plot_final_comparison(runs: dict[str, dict], figures_dir: Path) -> None:
    rows = []
    for name, run in runs.items():
        edf = run["evals"]
        if edf.empty:
            continue
        last = edf.iloc[-1]
        rows.append({
            "label": label_for(run["algorithm"], run["weighting"]),
            "algorithm": run["algorithm"],
            "weighting": run["weighting"],
            "greedy_accuracy": last.get("greedy_accuracy"),
            "pass@4": last.get("pass@4"),
        })
    if not rows:
        print("  (no final evals, skipping final_comparison)")
        return

    df = pd.DataFrame(rows).sort_values(["algorithm", "weighting"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in [
        ("greedy_accuracy", axes[0], "Greedy Accuracy (final)"),
        ("pass@4", axes[1], "Pass@4 (final)"),
    ]:
        bars = ax.bar(df["label"], df[metric],
                      color=[COLORS.get(w, "#999") for w in df["weighting"]],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, df[metric]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "final_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  final_comparison.png ({len(rows)} runs)")


def print_summary_table(runs: dict[str, dict]) -> None:
    rows = []
    for name, run in runs.items():
        row = {
            "run": name,
            "algorithm": run["algorithm"],
            "weighting": run["weighting"],
            "steps": int(run["train"]["step"].max()),
            "final_reward_mean": round(run["train"]["reward_mean"].iloc[-10:].mean(), 4),
        }
        edf = run["evals"]
        if not edf.empty:
            last = edf.iloc[-1]
            row["greedy_acc"] = last.get("greedy_accuracy")
            row["pass@4"] = last.get("pass@4")
        rows.append(row)
    if rows:
        table = pd.DataFrame(rows).sort_values(["algorithm", "weighting"])
        print("\n" + table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--figures-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()

    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {args.outputs_dir}")
    runs = discover_runs(args.outputs_dir)
    if not runs:
        print("No completed runs found.")
        return
    print(f"Found {len(runs)} runs with enough data\n")

    print("Generating figures:")
    plot_training_curves(runs, figures_dir)
    plot_eval_over_time(runs, figures_dir)
    plot_final_comparison(runs, figures_dir)
    print_summary_table(runs)
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()

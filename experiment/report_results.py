
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_num(x: float) -> str:
    if x == 0:
        return '0'
    if abs(x) < 1e-3:
        return f"{x:.1e}"
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize aggregate experiment results.')
    parser.add_argument('paths', nargs='+', help='aggregate_summary.json files')
    args = parser.parse_args()

    rows = [load_summary(Path(p)) for p in args.paths]
    rows.sort(key=lambda r: (r['task'], r['baseline'], r['weighting']))

    print('| Task | Baseline | Weighting | Greedy | Pass@k | Important Mass | Final Reward | Score Var |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        print(
            f"| {row['task']} | {row['baseline']} | {row['weighting']} | "
            f"{fmt_pct(row['greedy_accuracy']['mean'])} | "
            f"{fmt_pct(row['pass_at_k']['mean'])} | "
            f"{fmt_num(row['important_mass_mean']['mean'])} | "
            f"{fmt_num(row['final_reward_mean']['mean'])} | "
            f"{fmt_num(row['final_gradient_norm_variance']['mean'])} |"
        )


if __name__ == '__main__':
    main()

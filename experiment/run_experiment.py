
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.core.trainer import ExperimentRunner
from experiment.core.utils import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Run intrinsic token-weighting experiments.')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output-dir', default=None, help='Optional output directory override')
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    timestamp = dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    output_dir = args.output_dir or Path(__file__).resolve().parent / 'results' / f"{config_path.stem}-{timestamp}"
    output_dir = ensure_dir(output_dir)
    save_json(Path(output_dir) / 'resolved_config.json', config)
    runner = ExperimentRunner(config=config, output_dir=str(output_dir))
    summary = runner.run()
    print('=== aggregate summary ===')
    print(summary)


if __name__ == '__main__':
    main()

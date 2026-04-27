"""Pytest configuration for the experiment/hf test suite.

The package uses a flat layout (no top-level package), so we append the
parent directory to ``sys.path`` so tests can import modules such as
``token_weights`` and ``run_experiment`` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

HF_DIR = Path(__file__).resolve().parent.parent
if str(HF_DIR) not in sys.path:
    sys.path.insert(0, str(HF_DIR))

"""Run DARTS-inspired differentiable architecture search from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nas.darts_search import run_darts_search
from utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/darts_search.yaml",
        help="Path to the YAML DARTS config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    summary = run_darts_search(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

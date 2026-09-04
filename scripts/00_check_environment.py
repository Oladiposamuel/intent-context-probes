#!/usr/bin/env python3
"""Validate the Colab runtime before any expensive model execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config  # noqa: E402
from src.environment import build_environment_manifest, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Downgrade a missing GPU to a warning for local diagnostics only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path

    try:
        load_config(config_path)
        manifest, errors, warnings = build_environment_manifest(
            config_path,
            require_gpu=not args.allow_no_gpu,
        )
    except Exception as exc:
        print(f"Environment check could not run: {exc}", file=sys.stderr)
        return 1

    destination = REPOSITORY_ROOT / "artifacts/environment.json"
    write_json(destination, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nEnvironment manifest written to {destination}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors:
        print("\nEnvironment check FAILED.", file=sys.stderr)
        return 1
    print("\nEnvironment check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

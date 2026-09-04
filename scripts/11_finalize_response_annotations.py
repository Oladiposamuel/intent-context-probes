#!/usr/bin/env python3
"""Validate, lock, unblind, and analyze completed response annotations."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.environment import sync_directory_to_persistent
from src.response_annotations import finalize_annotations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "artifacts" / "annotations"
    parser.add_argument("--primary", type=Path, default=default / "completed_primary.csv")
    parser.add_argument("--repeat", type=Path, default=default / "completed_repeat.csv")
    parser.add_argument(
        "--identity-key", type=Path, default=default / "private_identity_key.csv"
    )
    parser.add_argument(
        "--repeat-key", type=Path, default=default / "private_repeat_key.csv"
    )
    return parser.parse_args()


args = parse_args()
config = load_config(ROOT / "configs/experiment.yaml")
output = ROOT / "artifacts" / "annotations"
summary = finalize_annotations(
    args.primary,
    args.repeat,
    args.identity_key,
    args.repeat_key,
    output_dir=output,
    iterations=config["evaluation"]["bootstrap_resamples"],
    seed=config["project"]["seed"],
)
sync_directory_to_persistent(output)
print(json.dumps(summary, indent=2))

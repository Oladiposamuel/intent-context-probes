#!/usr/bin/env python3
"""Prepare randomized blinded response-annotation sheets and private keys."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.environment import sync_directory_to_persistent
from src.response_annotations import prepare_annotation_package

config = load_config(ROOT / "configs/experiment.yaml")
prefixes = pd.read_parquet(ROOT / config["project"]["processed_path"])
dataset_hash = (ROOT / "data/FROZEN_DATASET.sha256").read_text().strip()
model_specs = {model["alias"]: model for model in config["models"]}
response_paths = {
    alias: ROOT / "artifacts" / "responses" / f"{alias}.jsonl"
    for alias in model_specs
}
output = ROOT / "artifacts" / "annotations"
manifest = prepare_annotation_package(
    prefixes,
    response_paths,
    model_specs,
    dataset_hash=dataset_hash,
    output_dir=output,
    seed=config["project"]["seed"],
)
sync_directory_to_persistent(output)
print(json.dumps(manifest, indent=2))

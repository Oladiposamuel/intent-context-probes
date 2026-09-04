#!/usr/bin/env python3
"""Run H6 shared-prefix checks and save out-of-fold Turns 1-4 trajectories."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.environment import sync_results_to_persistent
from src.probe_training import (
    fixed_outer_probe_trajectory,
    load_aligned_activations,
)
from src.trajectory import summarize_probe_trajectory

config = load_config(ROOT / "configs/experiment.yaml")
results = ROOT / "results"
results.mkdir(parents=True, exist_ok=True)
frame = pd.read_parquet(ROOT / config["project"]["processed_path"])
dataset_hash = (ROOT / "data/FROZEN_DATASET.sha256").read_text().strip()
probe_summary = json.loads((results / "probe_metrics.json").read_text())
summary = {}

for model in [item["alias"] for item in config["models"]]:
    method = f"{model}_full_history"
    layers, activations = load_aligned_activations(
        ROOT / f"artifacts/activations/{model}/full_history.npz",
        frame,
        dataset_hash,
    )
    trajectory = fixed_outer_probe_trajectory(
        frame,
        layers,
        activations,
        probe_summary[method]["selections"],
        config,
    )
    model_summary = summarize_probe_trajectory(trajectory)
    if not model_summary["early_prefix_checks"]["passed"]:
        raise RuntimeError(f"H6 shared-prefix sanity check failed for {model}.")
    trajectory.to_csv(results / f"{model}_trajectory_predictions.csv", index=False)
    summary[model] = model_summary

destination = results / "early_prefix_metrics.json"
destination.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
sync_results_to_persistent(results)
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.config import load_config
from src.environment import sync_results_to_persistent
from src.nested_cv import primary_metrics
from src.probe_training import load_aligned_activations, run_activation_probe

config = load_config(ROOT / "configs/experiment.yaml")
frame = pd.read_parquet(ROOT / config["project"]["processed_path"])
dataset_hash = (ROOT / "data/FROZEN_DATASET.sha256").read_text().strip()
out = ROOT / "results"
out.mkdir(exist_ok=True)
summary = {}
for model in [m["alias"] for m in config["models"]]:
    for mode in ["full_history", "current_message"]:
        layers, x = load_aligned_activations(
            ROOT / f"artifacts/activations/{model}/{mode}.npz", frame, dataset_hash
        )
        pred, selection = run_activation_probe(frame, layers, x, config)
        name = f"{model}_{mode}"
        pred.assign(method=name).to_csv(out / f"{name}_predictions.csv", index=False)
        summary[name] = {"metrics": primary_metrics(pred), "selections": selection}
(out / "probe_metrics.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
sync_results_to_persistent(out)
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.environment import sync_results_to_persistent
from src.nested_cv import primary_metrics
from src.text_baselines import run_length_control, run_text_baseline

config = load_config(ROOT / "configs/experiment.yaml")
frame = pd.read_parquet(ROOT / config["project"]["processed_path"])
out = ROOT / "results"
out.mkdir(exist_ok=True)

for mode, column in [
    ("full_history", "full_token_count"),
    ("current_message", "current_token_count"),
]:
    artifact = ROOT / f"artifacts/activations/qwen3_4b/{mode}.npz"
    with np.load(artifact, allow_pickle=False) as arrays:
        if arrays["example_ids"].astype(str).tolist() != frame.example_id.tolist():
            raise RuntimeError("Token-count artifact IDs do not align.")
        frame[column] = arrays["input_token_counts"].astype(int)

methods = {
    "tfidf_full": run_text_baseline(frame, "full_text_plain", config),
    "tfidf_current": run_text_baseline(frame, "current_user_message", config),
    "length": run_length_control(frame, config),
}
summary = {}
for name, (predictions, selections) in methods.items():
    predictions.assign(method=name).to_csv(
        out / f"{name}_predictions_verified.csv", index=False
    )
    summary[name] = {
        "metrics": primary_metrics(predictions),
        "selections": selections,
    }

(out / "baseline_metrics_verified.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
sync_results_to_persistent(out)
print(json.dumps(summary, indent=2))

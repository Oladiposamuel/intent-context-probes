#!/usr/bin/env python3
"""Evaluate saved full-history and current-message prompted judgements."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.environment import sync_results_to_persistent
from src.judgement_evaluation import (
    evaluate_prompted_judgements,
    load_prompted_judgement_frame,
)

config = load_config(ROOT / "configs/experiment.yaml")
prefixes = pd.read_parquet(ROOT / config["project"]["processed_path"])
dataset_hash = (ROOT / "data/FROZEN_DATASET.sha256").read_text().strip()
results = ROOT / "results"
results.mkdir(parents=True, exist_ok=True)
summary = {}

for model in config["models"]:
    alias = model["alias"]
    for mode in ("full_history", "current_message"):
        artifact = (
            ROOT / "artifacts" / "prompted_judgements" / f"{alias}_{mode}.jsonl"
        )
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing prompted-judgement artifact: {artifact}"
            )
        frame = load_prompted_judgement_frame(
            artifact,
            prefixes,
            model_alias=alias,
            model_revision=model["revision"],
            context_mode=mode,
            dataset_hash=dataset_hash,
        )
        method = f"{alias}_{mode}"
        summary[method] = evaluate_prompted_judgements(
            frame,
            context_mode=mode,
            iterations=config["evaluation"]["bootstrap_resamples"],
            seed=config["project"]["seed"],
        )
        frame.to_csv(results / f"{method}_prompted_judge_scores.csv", index=False)

destination = results / "prompted_judgement_metrics.json"
destination.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
sync_results_to_persistent(results)
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Independently verify prediction tables and compute paired uncertainty."""

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
from src.probe_training import fixed_outer_probe_predictions, load_aligned_activations
from src.statistics import (
    current_message_pairs_equal,
    paired_gap_summary,
    paired_method_bootstrap,
    paired_permuted_labels,
    stratified_paired_bootstrap,
    validate_predictions,
)

results = ROOT / "results"
selected_files = {}
for path in sorted(results.glob("*_predictions.csv")):
    method = path.stem.removesuffix("_predictions")
    selected_files[method] = path
for path in sorted(results.glob("*_predictions_verified.csv")):
    method = path.stem.removesuffix("_predictions_verified")
    selected_files[method] = path
if not selected_files:
    raise RuntimeError("No prediction tables found. Run scripts 05 and 06 first.")

summary = {}
prediction_tables = {}
for method, path in sorted(selected_files.items()):
    predictions = pd.read_csv(path)
    validate_predictions(predictions)
    prediction_tables[method] = predictions
    current = "current_message" in method or method == "tfidf_current"
    summary[method] = {
        "metrics_recomputed": primary_metrics(predictions),
        "paired_gaps": paired_gap_summary(predictions),
        "bootstrap": stratified_paired_bootstrap(predictions),
        "current_message_pair_scores_equal": (
            current_message_pairs_equal(predictions) if current else None
        ),
    }

# Registered minimum: 200 paired label permutations for both full-history probes.
config = load_config(ROOT / "configs/experiment.yaml")
frame = pd.read_parquet(ROOT / config["project"]["processed_path"])
dataset_hash = (ROOT / "data/FROZEN_DATASET.sha256").read_text().strip()
probe_summary = json.loads((results / "probe_metrics.json").read_text())
rng = np.random.default_rng(config["project"]["seed"])
for model in [item["alias"] for item in config["models"]]:
    method = f"{model}_full_history"
    layers, activations = load_aligned_activations(
        ROOT / f"artifacts/activations/{model}/full_history.npz",
        frame,
        dataset_hash,
    )
    observed = summary[method]["metrics_recomputed"]["macro_domain_auroc"]
    null = []
    for _ in range(200):
        labels = paired_permuted_labels(frame, rng)
        permuted = fixed_outer_probe_predictions(
            frame,
            layers,
            activations,
            probe_summary[method]["selections"],
            config,
            labels,
        )
        null.append(primary_metrics(permuted)["macro_domain_auroc"])
    summary[method]["paired_permutation"] = {
        "observed": observed,
        "iterations": 200,
        "seed": config["project"]["seed"],
        "p_value_greater_equal": float(
            (1 + np.count_nonzero(np.asarray(null) >= observed)) / 201
        ),
        "null_mean": float(np.mean(null)),
    }

comparisons = {}
comparison_specs = {
    "qwen_context_full_minus_current": (
        "qwen3_4b_full_history",
        "qwen3_4b_current_message",
    ),
    "saferl_context_full_minus_current": (
        "qwen3_4b_saferl_full_history",
        "qwen3_4b_saferl_current_message",
    ),
    "saferl_minus_qwen_full_history": (
        "qwen3_4b_saferl_full_history",
        "qwen3_4b_full_history",
    ),
    "qwen_activation_minus_tfidf_full": (
        "qwen3_4b_full_history",
        "tfidf_full",
    ),
    "saferl_activation_minus_tfidf_full": (
        "qwen3_4b_saferl_full_history",
        "tfidf_full",
    ),
    "qwen_activation_minus_length": (
        "qwen3_4b_full_history",
        "length",
    ),
    "saferl_activation_minus_length": (
        "qwen3_4b_saferl_full_history",
        "length",
    ),
}
for name, (first_method, second_method) in comparison_specs.items():
    comparisons[name] = paired_method_bootstrap(
        prediction_tables[first_method],
        prediction_tables[second_method],
        first_name=first_method,
        second_name=second_method,
        iterations=2000,
        seed=config["project"]["seed"],
    )
summary["method_comparisons"] = comparisons

destination = results / "verification_metrics.json"
destination.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
sync_results_to_persistent(results)
print(json.dumps(summary, indent=2))

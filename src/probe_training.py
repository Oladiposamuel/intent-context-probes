"""Nested-domain linear probes over saved activation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .nested_cv import run_nested_cv


def load_aligned_activations(path, frame, dataset_hash):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["dataset_hash"] != dataset_hash or not meta["complete"]:
        raise RuntimeError("Activation metadata is incompatible.")
    with np.load(path, allow_pickle=False) as a:
        ids = a["example_ids"].astype(str).tolist()
        layers = a["layers"].astype(int).tolist()
        x = a["X"].copy()
    if ids != frame.example_id.tolist():
        raise RuntimeError("Activation IDs do not align with prefixes.")
    return layers, x


def run_activation_probe(frame, layers, x, config):
    indexed = frame.reset_index(drop=True).copy()
    indexed["_row"] = np.arange(len(indexed))
    candidates = [
        (float(c), float(layer)) for layer in layers for c in config["probe"]["C_grid"]
    ]
    seed = config["project"]["seed"]

    def build(candidate):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=candidate[0],
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def features(rows):
        layer = int(rows.attrs.get("candidate_layer", layers[0]))
        return x[rows._row.to_numpy(), layers.index(layer), :]

    # Bind the selected layer into each model/feature closure.
    def factory(candidate):
        model = build(candidate)
        model._probe_layer = int(candidate[1])
        return model

    def feature(rows):
        # nested_cv calls feature after factory; layer is supplied via a temporary global.
        return x[rows._row.to_numpy(), layers.index(_ACTIVE[0]), :]

    _ACTIVE = [layers[0]]

    def wrapped(candidate):
        _ACTIVE[0] = int(candidate[1])
        return factory(candidate)

    return run_nested_cv(indexed, candidates, wrapped, feature, config["domains"])


def fixed_outer_probe_predictions(frame, layers, x, selections, config, labels=None):
    """Refit each outer fold with its nested-CV choice, optionally using permuted labels."""

    indexed = frame.reset_index(drop=True).copy()
    indexed["_row"] = np.arange(len(indexed))
    if labels is not None:
        indexed["binary_target"] = np.asarray(labels, dtype=float)
    eligible = indexed[indexed.turn_index.isin([3, 4])]
    if eligible.binary_target.isna().any():
        raise ValueError("Turns 3-4 must have complete binary labels.")
    selected = {item["test_domain"]: item["candidate"] for item in selections}
    predictions = []
    for test_domain in config["domains"]:
        c_value, layer = selected[test_domain]
        layer_index = layers.index(int(layer))
        train = eligible[eligible.domain != test_domain]
        test = eligible[eligible.domain == test_domain]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c_value),
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=config["project"]["seed"],
                    ),
                ),
            ]
        )
        model.fit(
            x[train._row.to_numpy(), layer_index, :],
            train.binary_target.astype(int),
        )
        scores = model.predict_proba(x[test._row.to_numpy(), layer_index, :])[:, 1]
        for (_, row), score in zip(test.iterrows(), scores):
            predictions.append(
                {
                    "example_id": row.example_id,
                    "scenario_id": row.scenario_id,
                    "domain": row.domain,
                    "branch": row.branch,
                    "turn_index": int(row.turn_index),
                    "binary_target": int(row.binary_target),
                    "score": float(score),
                    "test_domain": test_domain,
                }
            )
    import pandas as pd

    return pd.DataFrame(predictions)

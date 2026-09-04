"""Integrity checks and paired uncertainty estimates for Turn-4 predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .nested_cv import primary_metrics

REQUIRED_COLUMNS = {
    "example_id",
    "scenario_id",
    "domain",
    "branch",
    "turn_index",
    "binary_target",
    "score",
    "test_domain",
}


def validate_predictions(predictions: pd.DataFrame, expected_rows: int = 128) -> None:
    """Raise when an outer-fold prediction table is incomplete or contaminated."""

    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions, found {len(predictions)}."
        )
    if not predictions.example_id.is_unique:
        raise ValueError("Prediction example IDs are not unique.")
    if not (predictions.domain == predictions.test_domain).all():
        raise ValueError("A prediction was produced outside its held-out domain.")
    turn4 = predictions[predictions.turn_index == 4]
    if len(turn4) != expected_rows // 2:
        raise ValueError("Turn-4 prediction count is incomplete.")
    counts = turn4.groupby("domain").binary_target.nunique()
    if not (counts == 2).all():
        raise ValueError("Every domain must contain both Turn-4 labels.")


def current_message_pairs_equal(
    predictions: pd.DataFrame, *, atol: float = 1e-12
) -> bool:
    """Check that paired Turn-4 scores match when the rendered messages match."""

    turn4 = predictions[predictions.turn_index == 4]
    pivot = turn4.pivot(index="scenario_id", columns="branch", values="score")
    if not {"benign", "suspicious"}.issubset(pivot.columns):
        return False
    return bool(np.allclose(pivot.benign, pivot.suspicious, rtol=0.0, atol=atol))


def paired_gap_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Summarize suspicious-minus-benign Turn-4 score gaps by scenario."""

    turn4 = predictions[predictions.turn_index == 4]
    pivot = turn4.pivot(index="scenario_id", columns="branch", values="score")
    gaps = pivot.suspicious - pivot.benign
    return {
        "n_pairs": len(gaps),
        "mean_gap": float(gaps.mean()),
        "median_gap": float(gaps.median()),
        "fraction_positive": float((gaps > 0).mean()),
        "fraction_tied": float(np.isclose(gaps, 0.0, atol=1e-12).mean()),
    }


def stratified_paired_bootstrap(
    predictions: pd.DataFrame, *, iterations: int = 2000, seed: int = 42
) -> dict[str, float | int]:
    """Bootstrap paired scenarios within domains for macro-domain Turn-4 AUROC."""

    turn4 = predictions[predictions.turn_index == 4]
    rng = np.random.default_rng(seed)
    domains = sorted(turn4.domain.unique())
    samples: list[float] = []
    for _ in range(iterations):
        domain_aucs = []
        for domain in domains:
            block = turn4[turn4.domain == domain]
            scenario_ids = block.scenario_id.unique()
            chosen = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
            sampled = pd.concat(
                [block[block.scenario_id == scenario] for scenario in chosen],
                ignore_index=True,
            )
            domain_aucs.append(roc_auc_score(sampled.binary_target, sampled.score))
        samples.append(float(np.mean(domain_aucs)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": primary_metrics(predictions)["macro_domain_auroc"],
        "ci_low": float(low),
        "ci_high": float(high),
        "iterations": iterations,
        "seed": seed,
    }


def paired_permuted_labels(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Flip labels consistently for both turns of randomly selected scenario pairs."""

    labels = frame.binary_target.to_numpy(dtype=float, na_value=np.nan).copy()
    flipped = set(
        frame.scenario_id.unique()[
            rng.integers(0, 2, size=frame.scenario_id.nunique()).astype(bool)
        ]
    )
    mask = frame.scenario_id.isin(flipped).to_numpy() & ~np.isnan(labels)
    labels[mask] = 1 - labels[mask]
    return labels

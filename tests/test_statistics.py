import pandas as pd
import pytest

from src.probe_training import fixed_outer_probe_predictions
from src.statistics import (
    current_message_pairs_equal,
    paired_gap_summary,
    paired_method_bootstrap,
    paired_permuted_labels,
    stratified_paired_bootstrap,
    validate_predictions,
)


def prediction_frame():
    rows = []
    for domain_index, domain in enumerate(["a", "b", "c", "d"]):
        for scenario_index in range(8):
            scenario = f"{domain}-{scenario_index}"
            for turn in (3, 4):
                for branch, target in (("benign", 0), ("suspicious", 1)):
                    score = float(target) if turn == 4 else 0.5
                    rows.append(
                        {
                            "example_id": f"{scenario}-{branch}-{turn}",
                            "scenario_id": scenario,
                            "domain": domain,
                            "branch": branch,
                            "turn_index": turn,
                            "binary_target": target,
                            "score": score,
                            "test_domain": domain,
                        }
                    )
    return pd.DataFrame(rows)


def test_integrity_and_paired_statistics():
    frame = prediction_frame()
    validate_predictions(frame)
    assert not current_message_pairs_equal(frame)
    assert paired_gap_summary(frame)["fraction_positive"] == 1.0
    interval = stratified_paired_bootstrap(frame, iterations=20)
    assert interval["estimate"] == interval["ci_low"] == interval["ci_high"] == 1.0


def test_current_pairs_equal_and_integrity_failure():
    frame = prediction_frame()
    turn4 = frame.turn_index == 4
    frame.loc[turn4, "score"] = 0.5
    assert current_message_pairs_equal(frame)
    with pytest.raises(ValueError, match="unique"):
        validate_predictions(pd.concat([frame.iloc[:-1], frame.iloc[[0]]]))


def test_permutation_flips_whole_scenario_pairs():
    frame = prediction_frame()
    labels = paired_permuted_labels(frame, __import__("numpy").random.default_rng(42))
    changed = frame.assign(permuted=labels)
    by_scenario = changed.groupby("scenario_id").apply(
        lambda block: (block.binary_target != block.permuted).nunique(),
        include_groups=False,
    )
    assert (by_scenario == 1).all()


def test_permutation_preserves_unlabelled_early_turns():
    frame = prediction_frame()
    early = frame.iloc[:2].copy()
    early["turn_index"] = [1, 2]
    early["binary_target"] = pd.NA
    combined = pd.concat([early, frame], ignore_index=True)
    labels = paired_permuted_labels(
        combined, __import__("numpy").random.default_rng(42)
    )
    assert __import__("numpy").isnan(labels[:2]).all()
    assert not __import__("numpy").isnan(labels[2:]).any()

    selections = [
        {"test_domain": domain, "candidate": [1.0, 1.0]}
        for domain in ["a", "b", "c", "d"]
    ]
    activations = __import__("numpy").column_stack(
        [__import__("numpy").nan_to_num(labels), __import__("numpy").ones(len(labels))]
    )[:, None, :]
    predictions = fixed_outer_probe_predictions(
        combined,
        [1],
        activations,
        selections,
        {"domains": ["a", "b", "c", "d"], "project": {"seed": 42}},
        labels,
    )
    assert len(predictions) == 128


def test_paired_method_bootstrap_uses_shared_resamples():
    first = prediction_frame()
    second = first.copy()
    second.loc[second.turn_index == 4, "score"] = 0.5
    comparison = paired_method_bootstrap(
        first,
        second,
        first_name="perfect",
        second_name="chance",
        iterations=20,
        seed=42,
    )
    assert comparison["first_estimate"] == 1.0
    assert comparison["second_estimate"] == 0.5
    assert comparison["estimate_difference"] == 0.5
    assert comparison["ci_low"] == comparison["ci_high"] == 0.5
    assert comparison["bootstrap_fraction_greater_than_zero"] == 1.0

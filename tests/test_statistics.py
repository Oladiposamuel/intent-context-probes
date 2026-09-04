import pandas as pd
import pytest

from src.statistics import (
    current_message_pairs_equal,
    paired_gap_summary,
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

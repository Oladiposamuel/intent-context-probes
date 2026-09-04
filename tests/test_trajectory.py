import numpy as np
import pandas as pd

from src.probe_training import fixed_outer_probe_trajectory
from src.trajectory import summarize_probe_trajectory


def synthetic_prefixes():
    rows = []
    vectors = []
    for domain_index, domain in enumerate(["a", "b", "c", "d"]):
        for scenario_index in range(8):
            scenario = f"{domain}-{scenario_index}"
            shared = np.array([0.05 * scenario_index, 0.1 * domain_index])
            for turn in (1, 2, 3, 4):
                for branch in ("benign", "suspicious"):
                    target = int(branch == "suspicious")
                    rows.append(
                        {
                            "example_id": f"{scenario}-{branch}-{turn}",
                            "scenario_id": scenario,
                            "domain": domain,
                            "branch": branch,
                            "turn_index": turn,
                            "binary_target": target if turn in (3, 4) else pd.NA,
                        }
                    )
                    if turn in (1, 2):
                        vector = shared
                    else:
                        vector = np.array(
                            [2.0 if target else -2.0, 0.1 * domain_index]
                        )
                    vectors.append([vector])
    return pd.DataFrame(rows), np.asarray(vectors)


def test_trajectory_scores_shared_prefixes_without_training_on_them():
    frame, activations = synthetic_prefixes()
    selections = [
        {"test_domain": domain, "candidate": [1.0, 1.0]}
        for domain in ["a", "b", "c", "d"]
    ]
    config = {
        "domains": ["a", "b", "c", "d"],
        "project": {"seed": 42},
    }
    trajectory = fixed_outer_probe_trajectory(
        frame, [1], activations, selections, config
    )
    summary = summarize_probe_trajectory(trajectory)
    assert len(trajectory) == 256
    assert summary["early_prefix_checks"]["passed"]
    assert summary["turns"]["1"]["macro_domain_auroc"] == 0.5
    assert summary["turns"]["2"]["max_absolute_paired_gap"] == 0.0
    assert summary["turns"]["3"]["fraction_positive"] == 1.0


def test_trajectory_summary_detects_early_pair_mismatch():
    frame, activations = synthetic_prefixes()
    selections = [
        {"test_domain": domain, "candidate": [1.0, 1.0]}
        for domain in ["a", "b", "c", "d"]
    ]
    trajectory = fixed_outer_probe_trajectory(
        frame,
        [1],
        activations,
        selections,
        {"domains": ["a", "b", "c", "d"], "project": {"seed": 42}},
    )
    early_suspicious = (
        (trajectory.turn_index == 1) & (trajectory.branch == "suspicious")
    )
    trajectory.loc[early_suspicious, "score"] += 0.01
    summary = summarize_probe_trajectory(trajectory)
    assert not summary["early_prefix_checks"]["passed"]

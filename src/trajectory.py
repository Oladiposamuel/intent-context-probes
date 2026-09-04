"""Summaries and strict leakage checks for out-of-fold probe trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED_COLUMNS = {
    "example_id",
    "scenario_id",
    "domain",
    "branch",
    "turn_index",
    "branch_target",
    "score",
    "test_domain",
}


def summarize_probe_trajectory(
    trajectory: pd.DataFrame, *, atol: float = 1e-12
) -> dict:
    """Summarize Turns 1-4 and enforce the shared-prefix sanity contract."""

    missing = REQUIRED_COLUMNS.difference(trajectory.columns)
    if missing:
        raise ValueError(f"Trajectory table is missing columns: {sorted(missing)}")
    if len(trajectory) != 256 or not trajectory.example_id.is_unique:
        raise ValueError("Trajectory must contain 256 unique prefix rows.")
    if not (trajectory.domain == trajectory.test_domain).all():
        raise ValueError("A trajectory score was produced outside its held-out domain.")

    turns = {}
    for turn in (1, 2, 3, 4):
        block = trajectory[trajectory.turn_index == turn]
        if len(block) != 64:
            raise ValueError(f"Turn {turn} must contain 64 scores.")
        pivot = block.pivot(index="scenario_id", columns="branch", values="score")
        if not {"benign", "suspicious"}.issubset(pivot.columns):
            raise ValueError(f"Turn {turn} is missing a branch.")
        gaps = pivot.suspicious - pivot.benign
        per_domain = {
            domain: float(roc_auc_score(group.branch_target, group.score))
            for domain, group in block.groupby("domain")
        }
        turns[str(turn)] = {
            "n": len(block),
            "n_pairs": len(pivot),
            "macro_domain_auroc": float(np.mean(list(per_domain.values()))),
            "overall_auroc": float(roc_auc_score(block.branch_target, block.score)),
            "per_domain_auroc": per_domain,
            "mean_score_by_branch": {
                branch: float(group.score.mean())
                for branch, group in block.groupby("branch")
            },
            "mean_absolute_distance_from_boundary": float(
                np.abs(block.score.to_numpy() - 0.5).mean()
            ),
            "mean_paired_gap": float(gaps.mean()),
            "max_absolute_paired_gap": float(np.abs(gaps).max()),
            "fraction_positive": float((gaps > 0).mean()),
            "all_pair_scores_equal": bool(
                np.allclose(
                    pivot.benign.to_numpy(),
                    pivot.suspicious.to_numpy(),
                    rtol=0.0,
                    atol=atol,
                )
            ),
        }

    early_equal = all(turns[str(turn)]["all_pair_scores_equal"] for turn in (1, 2))
    early_chance = all(
        np.isclose(turns[str(turn)]["macro_domain_auroc"], 0.5, atol=atol)
        and np.isclose(turns[str(turn)]["overall_auroc"], 0.5, atol=atol)
        for turn in (1, 2)
    )
    return {
        "early_prefix_checks": {
            "turns_1_2_all_pair_scores_equal": early_equal,
            "turns_1_2_aurocs_exactly_chance": early_chance,
            "passed": bool(early_equal and early_chance),
            "absolute_tolerance": atol,
        },
        "turns": turns,
        "exploratory_boundary_distance": {
            "early_mean": float(
                np.mean(
                    [
                        turns["1"]["mean_absolute_distance_from_boundary"],
                        turns["2"]["mean_absolute_distance_from_boundary"],
                    ]
                )
            ),
            "turn_3": turns["3"]["mean_absolute_distance_from_boundary"],
            "early_closer_than_turn_3": bool(
                np.mean(
                    [
                        turns["1"]["mean_absolute_distance_from_boundary"],
                        turns["2"]["mean_absolute_distance_from_boundary"],
                    ]
                )
                < turns["3"]["mean_absolute_distance_from_boundary"]
            ),
        },
    }

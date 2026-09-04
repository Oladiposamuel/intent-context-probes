import json

import pandas as pd
import pytest

from src.judgement_evaluation import (
    evaluate_prompted_judgements,
    load_prompted_judgement_frame,
)


def judgement_frame(*, identical=False, missing=False):
    rows = []
    for domain in ["a", "b", "c", "d"]:
        for index in range(8):
            scenario = f"{domain}-{index}"
            shared_score = index / 10
            for branch, target in [("benign", 0), ("suspicious", 1)]:
                score = shared_score if identical else (0.2 if target == 0 else 0.8)
                if missing and domain == "a" and index == 0 and target == 1:
                    score = float("nan")
                rows.append(
                    {
                        "judgement_id": f"judge-{scenario}-{branch}",
                        "example_id": f"{scenario}-{branch}-4",
                        "scenario_id": scenario,
                        "domain": domain,
                        "branch": branch,
                        "binary_target": target,
                        "score": score,
                        "judge_label": (
                            "benign_evidence" if target == 0 else "suspicious_evidence"
                        ),
                        "parsed": not pd.isna(score),
                        "repair_attempted": False,
                        "parse_error": None if not pd.isna(score) else "failed",
                    }
                )
    return pd.DataFrame(rows)


def test_evaluation_reports_available_rows_and_complete_pairs():
    summary = evaluate_prompted_judgements(
        judgement_frame(missing=True),
        context_mode="full_history",
        iterations=20,
    )
    assert summary["parse_summary"]["parsed"] == 63
    assert summary["available_case_metrics"]["n"] == 63
    assert summary["complete_pair_metrics"]["n_pairs"] == 31
    assert summary["paired_bootstrap"]["estimate"] == 1.0


def test_current_message_identical_pairs_are_chance():
    summary = evaluate_prompted_judgements(
        judgement_frame(identical=True),
        context_mode="current_message",
        iterations=20,
    )
    assert summary["available_case_metrics"]["macro_domain_auroc"] == 0.5
    assert summary["complete_pair_metrics"]["fraction_tied"] == 1.0
    assert summary["current_message_pair_scores_equal"]


def test_loader_validates_metadata_and_preserves_parse_failure(tmp_path):
    prefixes = judgement_frame()[[
        "example_id", "scenario_id", "domain", "branch", "binary_target"
    ]].copy()
    prefixes["is_primary_t4"] = True
    records = []
    for index, row in prefixes.iterrows():
        parsed = {
            "score": 20 if row.branch == "benign" else 80,
            "label": (
                "benign_evidence"
                if row.branch == "benign"
                else "suspicious_evidence"
            ),
            "brief_reason": "test",
        }
        if index == 0:
            parsed = None
        records.append(
            {
                "judgement_id": f"model__full_history__{row.example_id}",
                "example_id": row.example_id,
                "model_revision": "revision",
                "dataset_hash": "hash",
                "context_mode": "full_history",
                "parsed": parsed,
                "repair_output": "repair" if parsed is None else None,
                "parse_error": "failed" if parsed is None else None,
            }
        )
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    loaded = load_prompted_judgement_frame(
        path,
        prefixes,
        model_alias="model",
        model_revision="revision",
        context_mode="full_history",
        dataset_hash="hash",
    )
    assert len(loaded) == 64
    assert loaded.parsed.sum() == 63
    assert loaded.repair_attempted.sum() == 1

    records[1]["model_revision"] = "wrong"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision"):
        load_prompted_judgement_frame(
            path,
            prefixes,
            model_alias="model",
            model_revision="revision",
            context_mode="full_history",
            dataset_hash="hash",
        )

"""Evaluation of fixed prompted judgements with explicit missing-data handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_prompted_judgement_frame(
    path: str | Path,
    prefixes: pd.DataFrame,
    *,
    model_alias: str,
    model_revision: str,
    context_mode: str,
    dataset_hash: str,
) -> pd.DataFrame:
    """Validate a judgement JSONL artifact and align it to frozen Turn-4 rows."""

    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 64:
        raise ValueError(f"Expected 64 judgement records, found {len(records)}.")
    judgement_ids = [record.get("judgement_id") for record in records]
    if len(set(judgement_ids)) != 64:
        raise ValueError("Judgement IDs are not unique.")

    primary = prefixes[prefixes["is_primary_t4"]].copy()
    expected_ids = set(primary.example_id)
    actual_ids = {record.get("example_id") for record in records}
    if actual_ids != expected_ids:
        raise ValueError("Judgement example IDs do not match frozen Turn-4 rows.")

    by_id = {record["example_id"]: record for record in records}
    rows = []
    for _, prefix in primary.iterrows():
        record = by_id[prefix.example_id]
        if record.get("model_revision") != model_revision:
            raise ValueError("Judgement model revision is incompatible.")
        if record.get("dataset_hash") != dataset_hash:
            raise ValueError("Judgement dataset hash is incompatible.")
        if record.get("context_mode") != context_mode:
            raise ValueError("Judgement context mode is incompatible.")
        expected_prefix = f"{model_alias}__{context_mode}__"
        if not record["judgement_id"].startswith(expected_prefix):
            raise ValueError("Judgement ID has an incompatible model or context prefix.")

        parsed = record.get("parsed")
        if parsed is None:
            score = np.nan
            judge_label = None
        else:
            score = parsed.get("score")
            judge_label = parsed.get("label")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 100
            ):
                raise ValueError("Parsed judgement score is outside 0-100.")
            if judge_label not in {
                "benign_evidence",
                "underdetermined",
                "suspicious_evidence",
            }:
                raise ValueError("Parsed judgement label is invalid.")
            score = float(score) / 100.0

        rows.append(
            {
                "judgement_id": record["judgement_id"],
                "example_id": prefix.example_id,
                "scenario_id": prefix.scenario_id,
                "domain": prefix.domain,
                "branch": prefix.branch,
                "binary_target": int(prefix.binary_target),
                "score": score,
                "judge_label": judge_label,
                "parsed": parsed is not None,
                "repair_attempted": record.get("repair_output") is not None,
                "parse_error": record.get("parse_error"),
            }
        )
    return pd.DataFrame(rows)


def _macro_domain_auc(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    per_domain = {}
    for domain, block in frame.groupby("domain"):
        if block.binary_target.nunique() != 2:
            raise ValueError(f"Domain {domain} does not contain both labels.")
        per_domain[domain] = float(roc_auc_score(block.binary_target, block.score))
    return float(np.mean(list(per_domain.values()))), per_domain


def _paired_bootstrap(
    complete: pd.DataFrame, *, iterations: int = 2000, seed: int = 42
) -> dict:
    estimate, _ = _macro_domain_auc(complete)
    rng = np.random.default_rng(seed)
    domains = sorted(complete.domain.unique())
    samples = []
    for _ in range(iterations):
        blocks = []
        for domain in domains:
            domain_block = complete[complete.domain == domain]
            scenario_ids = domain_block.scenario_id.unique()
            chosen = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
            blocks.extend(
                domain_block[domain_block.scenario_id == scenario]
                for scenario in chosen
            )
        sampled = pd.concat(blocks, ignore_index=True)
        value, _ = _macro_domain_auc(sampled)
        samples.append(value)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "iterations": iterations,
        "seed": seed,
    }


def evaluate_prompted_judgements(
    frame: pd.DataFrame,
    *,
    context_mode: str,
    iterations: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute available-case metrics and complete-pair uncertainty."""

    if len(frame) != 64 or not frame.example_id.is_unique:
        raise ValueError("Prompted judgement frame must contain 64 unique rows.")
    available = frame.dropna(subset=["score"]).copy()
    if available.empty:
        raise ValueError("No prompted judgements parsed successfully.")

    available_macro, available_per_domain = _macro_domain_auc(available)
    pivot = frame.pivot(index="scenario_id", columns="branch", values="score")
    complete_ids = pivot.dropna(subset=["benign", "suspicious"]).index
    complete = frame[
        frame.scenario_id.isin(complete_ids) & frame.score.notna()
    ].copy()
    complete_macro, complete_per_domain = _macro_domain_auc(complete)
    complete_pivot = pivot.loc[complete_ids]
    gaps = complete_pivot.suspicious - complete_pivot.benign

    current_equal = None
    if context_mode == "current_message":
        current_equal = bool(
            np.allclose(
                complete_pivot.benign,
                complete_pivot.suspicious,
                rtol=0.0,
                atol=1e-12,
            )
        )

    return {
        "parse_summary": {
            "total": len(frame),
            "parsed": int(frame.parsed.sum()),
            "failures": int((~frame.parsed).sum()),
            "repair_attempts": int(frame.repair_attempted.sum()),
        },
        "available_case_metrics": {
            "macro_domain_auroc": available_macro,
            "overall_auroc": float(
                roc_auc_score(available.binary_target, available.score)
            ),
            "n": len(available),
            "per_domain": available_per_domain,
        },
        "complete_pair_metrics": {
            "macro_domain_auroc": complete_macro,
            "overall_auroc": float(
                roc_auc_score(complete.binary_target, complete.score)
            ),
            "n_rows": len(complete),
            "n_pairs": len(complete_ids),
            "per_domain": complete_per_domain,
            "mean_paired_gap": float(gaps.mean()),
            "median_paired_gap": float(gaps.median()),
            "fraction_positive": float((gaps > 0).mean()),
            "fraction_tied": float(np.isclose(gaps, 0.0, atol=1e-12).mean()),
        },
        "paired_bootstrap": _paired_bootstrap(
            complete, iterations=iterations, seed=seed
        ),
        "current_message_pair_scores_equal": current_equal,
        "score_summary_by_branch": {
            branch: {
                "mean": float(block.score.mean()),
                "median": float(block.score.median()),
                "n": int(block.score.notna().sum()),
            }
            for branch, block in frame.groupby("branch")
        },
        "label_counts_by_branch": {
            branch: {
                str(label): int(count)
                for label, count in block.judge_label.fillna("missing")
                .value_counts()
                .items()
            }
            for branch, block in frame.groupby("branch")
        },
    }

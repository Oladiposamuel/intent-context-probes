"""Build auditable full-history and current-message prefix records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.rendering import plain_text_transcript, validate_messages
from src.schemas import ALLOWED_BRANCHES, binary_target


PREFIX_COLUMNS = (
    "example_id",
    "scenario_id",
    "domain",
    "subtopic",
    "difficulty",
    "branch",
    "turn_index",
    "context_label",
    "local_label",
    "binary_target",
    "messages_full_json",
    "current_user_message",
    "full_text_plain",
    "pair_turn_id",
    "is_primary_t4",
    "dataset_hash",
)


def build_full_messages(
    scenario: dict[str, Any],
    branch: str,
    turn_index: int,
) -> list[dict[str, str]]:
    """Return the controlled conversation prefix through one user turn."""

    if branch not in ALLOWED_BRANCHES:
        raise ValueError(f"Unknown branch: {branch!r}")
    if turn_index not in {1, 2, 3, 4}:
        raise ValueError("turn_index must be one of 1, 2, 3, or 4.")

    shared = scenario["shared"]
    messages: list[dict[str, str]] = [
        {"role": "user", "content": shared["user_turn_1"]},
    ]
    if turn_index >= 2:
        messages.extend(
            [
                {"role": "assistant", "content": shared["assistant_turn_1"]},
                {"role": "user", "content": shared["user_turn_2"]},
            ]
        )
    if turn_index >= 3:
        messages.extend(
            [
                {"role": "assistant", "content": shared["assistant_turn_2"]},
                {"role": "user", "content": scenario[branch]["user_turn_3"]},
            ]
        )
    if turn_index >= 4:
        messages.extend(
            [
                {"role": "assistant", "content": shared["assistant_turn_3"]},
                {"role": "user", "content": shared["user_turn_4"]},
            ]
        )

    validate_messages(messages)
    return messages


def get_current_user_message(
    scenario: dict[str, Any],
    branch: str,
    turn_index: int,
) -> str:
    """Return only the current user message for the context ablation."""

    if branch not in ALLOWED_BRANCHES:
        raise ValueError(f"Unknown branch: {branch!r}")
    shared = scenario["shared"]
    if turn_index == 1:
        return shared["user_turn_1"]
    if turn_index == 2:
        return shared["user_turn_2"]
    if turn_index == 3:
        return scenario[branch]["user_turn_3"]
    if turn_index == 4:
        return shared["user_turn_4"]
    raise ValueError("turn_index must be one of 1, 2, 3, or 4.")


def _labels_for(
    scenario: dict[str, Any],
    branch: str,
    turn_index: int,
) -> tuple[str, str]:
    if turn_index in {1, 2}:
        return "underdetermined", "underdetermined"
    context_label = scenario[branch][f"context_label_turn_{turn_index}"]
    local_label = context_label if turn_index == 3 else "underdetermined"
    return context_label, local_label


def flatten_scenarios(
    scenarios: list[dict[str, Any]],
    dataset_hash: str,
) -> pd.DataFrame:
    """Flatten canonical scenarios into one row per branch and user turn."""

    if not dataset_hash:
        raise ValueError("dataset_hash must be a non-empty string.")

    rows: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda item: item["scenario_id"]):
        for branch in ALLOWED_BRANCHES:
            for turn_index in range(1, 5):
                messages = build_full_messages(scenario, branch, turn_index)
                current_message = get_current_user_message(
                    scenario, branch, turn_index
                )
                context_label, local_label = _labels_for(
                    scenario, branch, turn_index
                )
                scenario_id = scenario["scenario_id"]
                rows.append(
                    {
                        "example_id": (
                            f"{scenario_id}__{branch}__t{turn_index}"
                        ),
                        "scenario_id": scenario_id,
                        "domain": scenario["domain"],
                        "subtopic": scenario["subtopic"],
                        "difficulty": scenario["difficulty"],
                        "branch": branch,
                        "turn_index": turn_index,
                        "context_label": context_label,
                        "local_label": local_label,
                        "binary_target": binary_target(context_label),
                        "messages_full_json": json.dumps(
                            messages,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "current_user_message": current_message,
                        "full_text_plain": plain_text_transcript(messages),
                        "pair_turn_id": f"{scenario_id}__t{turn_index}",
                        "is_primary_t4": turn_index == 4,
                        "dataset_hash": dataset_hash,
                    }
                )

    frame = pd.DataFrame(rows, columns=PREFIX_COLUMNS)
    if not frame.empty:
        frame["binary_target"] = pd.array(
            frame["binary_target"], dtype="Int64"
        )
    return frame


def validate_prefix_dataframe(
    frame: pd.DataFrame,
    scenario_count: int,
) -> list[str]:
    """Check row counts and all within-pair prefix invariants."""

    errors: list[str] = []
    missing_columns = sorted(set(PREFIX_COLUMNS) - set(frame.columns))
    if missing_columns:
        return [f"Prefix table is missing columns: {missing_columns}."]

    expected_rows = scenario_count * 2 * 4
    if len(frame) != expected_rows:
        errors.append(
            f"Expected {expected_rows} prefix rows; found {len(frame)}."
        )
    if frame["example_id"].duplicated().any():
        errors.append("Prefix example_id values must be unique.")
    if frame["pair_turn_id"].value_counts().ne(2).any():
        errors.append("Every pair_turn_id must contain exactly two branches.")
    if frame["dataset_hash"].nunique(dropna=False) != 1:
        errors.append("Every prefix row must carry the same dataset hash.")

    supervised = frame[frame["turn_index"].isin([3, 4])]
    if len(supervised) != scenario_count * 4:
        errors.append("The supervised Turn-3/Turn-4 row count is invalid.")
    primary = frame[frame["is_primary_t4"]]
    if len(primary) != scenario_count * 2:
        errors.append("The primary Turn-4 row count is invalid.")

    for scenario_id, scenario_rows in frame.groupby("scenario_id", sort=True):
        if set(scenario_rows["branch"]) != set(ALLOWED_BRANCHES):
            errors.append(f"{scenario_id}: both branches are required.")
            continue
        for turn_index in range(1, 5):
            pair = scenario_rows[scenario_rows["turn_index"] == turn_index]
            if len(pair) != 2:
                errors.append(
                    f"{scenario_id} Turn {turn_index}: expected two rows."
                )
                continue
            benign = pair[pair["branch"] == "benign"].iloc[0]
            suspicious = pair[pair["branch"] == "suspicious"].iloc[0]

            if turn_index in {1, 2}:
                if benign["messages_full_json"] != suspicious["messages_full_json"]:
                    errors.append(
                        f"{scenario_id} Turn {turn_index}: shared full prefixes differ."
                    )
                if benign["current_user_message"] != suspicious["current_user_message"]:
                    errors.append(
                        f"{scenario_id} Turn {turn_index}: shared current messages differ."
                    )
            if turn_index == 4:
                if benign["current_user_message"] != suspicious["current_user_message"]:
                    errors.append(
                        f"{scenario_id} Turn 4: current messages must be identical."
                    )
                benign_messages = json.loads(benign["messages_full_json"])
                suspicious_messages = json.loads(suspicious["messages_full_json"])
                differing = [
                    index
                    for index, values in enumerate(
                        zip(benign_messages, suspicious_messages)
                    )
                    if values[0] != values[1]
                ]
                if differing != [4]:
                    errors.append(
                        f"{scenario_id} Turn 4: full histories must differ "
                        "only at the Turn-3 user message."
                    )

    return errors


def write_prefix_outputs(
    frame: pd.DataFrame,
    parquet_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write Parquet and CSV atomically after all validation has passed."""

    parquet_path = Path(parquet_path)
    csv_path = Path(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = parquet_path.with_name(parquet_path.name + ".tmp")
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")

    try:
        frame.to_parquet(parquet_tmp, index=False, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "Writing prefixes.parquet requires pyarrow. Install requirements.txt."
        ) from exc
    frame.to_csv(csv_tmp, index=False, lineterminator="\n")
    parquet_tmp.replace(parquet_path)
    csv_tmp.replace(csv_path)

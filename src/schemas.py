"""Canonical schema values and runtime validation for scenario records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Literal


ContextLabel = Literal[
    "underdetermined",
    "benign_evidence",
    "suspicious_evidence",
]
Branch = Literal["benign", "suspicious"]
Difficulty = Literal["clear", "moderate", "subtle"]
DatasetStatus = Literal["draft", "frozen"]

SCHEMA_VERSION = "1.0"

ALLOWED_DOMAINS = (
    "cybersecurity",
    "biosecurity_public_health",
    "chemical_safety",
    "autonomous_physical_security",
)
ALLOWED_CONTEXT_LABELS = (
    "underdetermined",
    "benign_evidence",
    "suspicious_evidence",
)
ALLOWED_BRANCHES = ("benign", "suspicious")
ALLOWED_DIFFICULTIES = ("clear", "moderate", "subtle")
ALLOWED_DATASET_STATUSES = ("draft", "frozen")
ALLOWED_SOURCE_TYPES = ("newly_written", "adapted", "copied")

TOP_LEVEL_FIELDS = {
    "scenario_id",
    "domain",
    "subtopic",
    "difficulty",
    "source_type",
    "source_reference",
    "drafted_with_llm",
    "reviewer",
    "status",
    "publication_redaction_required",
    "shared",
    "benign",
    "suspicious",
    "audit",
}
SHARED_FIELDS = {
    "user_turn_1",
    "assistant_turn_1",
    "user_turn_2",
    "assistant_turn_2",
    "assistant_turn_3",
    "user_turn_4",
}
BRANCH_FIELDS = {
    "user_turn_3",
    "context_label_turn_3",
    "context_label_turn_4",
    "evidence_rationale",
}
AUDIT_FIELDS = {
    "turn_1_exact_match",
    "turn_2_exact_match",
    "turn_4_exact_match",
    "assistant_stubs_exact_match",
    "turn_3_word_count_ratio",
    "manual_label_confirmed",
    "notes",
}

_SCENARIO_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def binary_target(label: str) -> int | None:
    """Map the supervised evidence labels to the binary probe target."""

    if label == "underdetermined":
        return None
    if label == "benign_evidence":
        return 0
    if label == "suspicious_evidence":
        return 1
    raise ValueError(f"Unknown context label: {label!r}")


def _mapping_errors(
    value: object,
    expected_fields: set[str],
    path: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path} must be an object."]

    actual = set(value)
    errors: list[str] = []
    missing = sorted(expected_fields - actual)
    unknown = sorted(actual - expected_fields)
    if missing:
        errors.append(f"{path} is missing required fields: {missing}.")
    if unknown:
        errors.append(f"{path} contains unknown fields: {unknown}.")
    return errors


def _nonempty_string_error(value: object, path: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{path} must be a non-empty string."
    if "\x00" in value:
        return f"{path} must not contain a null character."
    return None


def validate_required_fields(scenario: object) -> list[str]:
    """Return schema/type errors for one canonical scenario dictionary.

    Validation deliberately rejects unknown fields. A misspelled field should
    not disappear silently when the dataset is normalized and frozen.
    """

    errors = _mapping_errors(scenario, TOP_LEVEL_FIELDS, "scenario")
    if not isinstance(scenario, Mapping):
        return errors

    scenario_id = scenario.get("scenario_id")
    error = _nonempty_string_error(scenario_id, "scenario.scenario_id")
    if error:
        errors.append(error)
    elif not _SCENARIO_ID_PATTERN.fullmatch(str(scenario_id)):
        errors.append(
            "scenario.scenario_id must match ^[a-z][a-z0-9_]*$."
        )

    for field in ("subtopic", "reviewer"):
        error = _nonempty_string_error(
            scenario.get(field), f"scenario.{field}"
        )
        if error:
            errors.append(error)

    if scenario.get("domain") not in ALLOWED_DOMAINS:
        errors.append(
            f"scenario.domain must be one of {list(ALLOWED_DOMAINS)}."
        )
    if scenario.get("difficulty") not in ALLOWED_DIFFICULTIES:
        errors.append(
            "scenario.difficulty must be one of "
            f"{list(ALLOWED_DIFFICULTIES)}."
        )
    if scenario.get("status") not in ALLOWED_DATASET_STATUSES:
        errors.append(
            f"scenario.status must be one of {list(ALLOWED_DATASET_STATUSES)}."
        )

    source_type = scenario.get("source_type")
    if source_type not in ALLOWED_SOURCE_TYPES:
        errors.append(
            f"scenario.source_type must be one of {list(ALLOWED_SOURCE_TYPES)}."
        )
    source_reference = scenario.get("source_reference")
    if source_reference is not None and (
        not isinstance(source_reference, str) or not source_reference.strip()
    ):
        errors.append(
            "scenario.source_reference must be null or a non-empty string."
        )
    if source_type in {"adapted", "copied"} and source_reference is None:
        errors.append(
            "scenario.source_reference is required for adapted or copied data."
        )

    for field in ("drafted_with_llm", "publication_redaction_required"):
        if type(scenario.get(field)) is not bool:
            errors.append(f"scenario.{field} must be a boolean.")

    shared = scenario.get("shared")
    errors.extend(_mapping_errors(shared, SHARED_FIELDS, "scenario.shared"))
    if isinstance(shared, Mapping):
        for field in sorted(SHARED_FIELDS):
            error = _nonempty_string_error(
                shared.get(field), f"scenario.shared.{field}"
            )
            if error:
                errors.append(error)

    for branch_name in ALLOWED_BRANCHES:
        branch = scenario.get(branch_name)
        path = f"scenario.{branch_name}"
        errors.extend(_mapping_errors(branch, BRANCH_FIELDS, path))
        if not isinstance(branch, Mapping):
            continue
        for field in ("user_turn_3", "evidence_rationale"):
            error = _nonempty_string_error(branch.get(field), f"{path}.{field}")
            if error:
                errors.append(error)
        for field in ("context_label_turn_3", "context_label_turn_4"):
            if branch.get(field) not in ALLOWED_CONTEXT_LABELS:
                errors.append(
                    f"{path}.{field} must be one of "
                    f"{list(ALLOWED_CONTEXT_LABELS)}."
                )

    audit = scenario.get("audit")
    errors.extend(_mapping_errors(audit, AUDIT_FIELDS, "scenario.audit"))
    if isinstance(audit, Mapping):
        for field in AUDIT_FIELDS - {"turn_3_word_count_ratio", "notes"}:
            if type(audit.get(field)) is not bool:
                errors.append(f"scenario.audit.{field} must be a boolean.")
        ratio = audit.get("turn_3_word_count_ratio")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or float(ratio) < 1.0
        ):
            errors.append(
                "scenario.audit.turn_3_word_count_ratio must be a finite "
                "number greater than or equal to 1."
            )
        if not isinstance(audit.get("notes"), str):
            errors.append("scenario.audit.notes must be a string.")

    return errors

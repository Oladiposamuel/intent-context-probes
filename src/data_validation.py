"""Dataset validation, audits, deterministic normalization, and freezing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.build_prefixes import build_full_messages
from src.rendering import plain_text_transcript
from src.schemas import (
    ALLOWED_CONTEXT_LABELS,
    ALLOWED_DIFFICULTIES,
    ALLOWED_DOMAINS,
    SCHEMA_VERSION,
    validate_required_fields,
)


EXPECTED_SCENARIO_COUNT = 32
EXPECTED_SCENARIOS_PER_DOMAIN = 8
MAX_FULL_HISTORY_WORDS = 512
REVIEW_LABELS = {
    "benign_evidence",
    "suspicious_evidence",
    "ambiguous",
}
EXPLICIT_LABEL_TERMS = {
    "attack",
    "authorized",
    "defensive",
    "ethical",
    "illegal",
    "malicious",
    "safety",
    "terrorist",
    "weapon",
}

PAIRWISE_REVIEW_COLUMNS = (
    "same_underlying_subject",
    "comparable_capability_level",
    "turn3_differs_mainly_in_use",
    "evidence_rationale_is_defensible",
    "turn4_is_locally_ambiguous",
    "not_based_on_identity_stereotype",
    "benign_branch_has_positive_evidence",
)

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


class DatasetValidationError(ValueError):
    """Raised when an invalid or unexpectedly changed dataset is used."""


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable validation result."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, Any]
    for_freeze: bool

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "for_freeze": self.for_freeze,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "counts": self.counts,
        }


def _words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]


def _word_count(text: str) -> int:
    return len(_words(text))


def turn3_word_count_ratio(scenario: dict[str, Any]) -> float:
    """Return the larger Turn-3 word count divided by the smaller count."""

    benign = _word_count(scenario["benign"]["user_turn_3"])
    suspicious = _word_count(scenario["suspicious"]["user_turn_3"])
    if min(benign, suspicious) == 0:
        return math.inf
    return max(benign, suspicious) / min(benign, suspicious)


def validate_scenario_structure(scenario: dict[str, Any]) -> list[str]:
    """Return structural errors for one schema-valid scenario."""

    errors = validate_required_fields(scenario)
    if errors:
        return errors

    if scenario["benign"]["user_turn_3"].strip() == scenario["suspicious"][
        "user_turn_3"
    ].strip():
        errors.append("Benign and suspicious user Turn 3 must differ.")

    audit = scenario["audit"]
    for field in (
        "turn_1_exact_match",
        "turn_2_exact_match",
        "turn_4_exact_match",
        "assistant_stubs_exact_match",
    ):
        if audit[field] is not True:
            errors.append(f"audit.{field} must be true.")

    actual_ratio = turn3_word_count_ratio(scenario)
    recorded_ratio = float(audit["turn_3_word_count_ratio"])
    if not math.isclose(actual_ratio, recorded_ratio, rel_tol=0.005, abs_tol=0.005):
        errors.append(
            "audit.turn_3_word_count_ratio does not match the computed "
            f"ratio ({actual_ratio:.4f})."
        )

    for branch in ("benign", "suspicious"):
        messages = build_full_messages(scenario, branch, 4)
        full_words = sum(_word_count(item["content"]) for item in messages)
        if full_words > MAX_FULL_HISTORY_WORDS:
            errors.append(
                f"{branch} full Turn-4 history has {full_words} words; "
                f"the pre-tokenization limit is {MAX_FULL_HISTORY_WORDS}."
            )

    return errors


def validate_pair_equality(scenario: dict[str, Any]) -> list[str]:
    """Return within-pair equality errors.

    The canonical schema stores shared turns only once. This function also
    constructs both branch histories to ensure the realized Turn-4 prefixes
    differ at exactly the branch-specific Turn-3 user message.
    """

    if validate_required_fields(scenario):
        return ["Pair equality cannot be checked until schema errors are fixed."]

    benign = build_full_messages(scenario, "benign", 4)
    suspicious = build_full_messages(scenario, "suspicious", 4)
    differing = [
        index
        for index, pair in enumerate(zip(benign, suspicious))
        if pair[0] != pair[1]
    ]
    if differing != [4]:
        return [
            "Turn-4 full histories must differ only at message index 4 "
            "(the user Turn-3 message)."
        ]
    return []


def validate_labels(scenario: dict[str, Any]) -> list[str]:
    """Enforce the frozen contextual label transitions."""

    if validate_required_fields(scenario):
        return ["Labels cannot be checked until schema errors are fixed."]

    errors: list[str] = []
    expected = {
        "benign": "benign_evidence",
        "suspicious": "suspicious_evidence",
    }
    for branch, label in expected.items():
        for turn in (3, 4):
            actual = scenario[branch][f"context_label_turn_{turn}"]
            if actual != label:
                errors.append(
                    f"{branch}.context_label_turn_{turn} must be {label!r}; "
                    f"found {actual!r}."
                )
    return errors


def _scenario_text_signature(scenario: dict[str, Any]) -> str:
    payload = {
        "shared": scenario["shared"],
        "benign_turn_3": scenario["benign"]["user_turn_3"],
        "suspicious_turn_3": scenario["suspicious"]["user_turn_3"],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_dataset(
    scenarios: list[dict[str, Any]],
    *,
    for_freeze: bool = False,
) -> ValidationReport:
    """Validate a partial draft or the complete dataset intended for freezing."""

    errors: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    text_signatures: dict[str, str] = {}
    domain_counts: Counter[str] = Counter()
    difficulty_counts: dict[str, Counter[str]] = {
        domain: Counter() for domain in ALLOWED_DOMAINS
    }
    turn4_counts: Counter[str] = Counter()

    if not scenarios:
        errors.append("The dataset contains no scenarios.")

    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = (
            scenario.get("scenario_id", f"line_{index}")
            if isinstance(scenario, dict)
            else f"line_{index}"
        )
        prefix = f"{scenario_id}: "
        schema_errors = validate_required_fields(scenario)
        errors.extend(prefix + error for error in schema_errors)
        if schema_errors:
            continue

        ids.append(scenario["scenario_id"])
        domain_counts[scenario["domain"]] += 1
        difficulty_counts[scenario["domain"]][scenario["difficulty"]] += 1
        turn4_counts[scenario["shared"]["user_turn_4"].strip()] += 1

        errors.extend(
            prefix + error for error in validate_scenario_structure(scenario)
        )
        errors.extend(prefix + error for error in validate_pair_equality(scenario))
        errors.extend(prefix + error for error in validate_labels(scenario))

        signature = _scenario_text_signature(scenario)
        if signature in text_signatures:
            errors.append(
                prefix
                + "duplicates all conversation text from scenario "
                + repr(text_signatures[signature])
                + "."
            )
        else:
            text_signatures[signature] = scenario["scenario_id"]

        ratio = turn3_word_count_ratio(scenario)
        if ratio > 1.25:
            warnings.append(
                prefix
                + f"Turn-3 word-count ratio is {ratio:.3f}, above the "
                "approximately 1.25 review target."
            )

        if for_freeze:
            if scenario["status"] != "frozen":
                errors.append(prefix + "status must be 'frozen' at freeze time.")
            if scenario["audit"]["manual_label_confirmed"] is not True:
                errors.append(
                    prefix + "audit.manual_label_confirmed must be true."
                )

    duplicate_ids = sorted(
        scenario_id for scenario_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"Duplicate scenario IDs: {duplicate_ids}.")

    repeated_turn4 = sorted(
        (text, count) for text, count in turn4_counts.items() if count > 2
    )
    for text, count in repeated_turn4:
        warnings.append(
            f"The same Turn-4 message is reused in {count} scenarios: {text!r}."
        )

    if for_freeze:
        if len(scenarios) != EXPECTED_SCENARIO_COUNT:
            errors.append(
                f"A frozen main dataset must contain exactly "
                f"{EXPECTED_SCENARIO_COUNT} scenarios; found {len(scenarios)}."
            )
        for domain in ALLOWED_DOMAINS:
            count = domain_counts[domain]
            if count != EXPECTED_SCENARIOS_PER_DOMAIN:
                errors.append(
                    f"Domain {domain!r} must contain exactly "
                    f"{EXPECTED_SCENARIOS_PER_DOMAIN} scenarios; found {count}."
                )
            expected_difficulties = {"clear": 2, "moderate": 4, "subtle": 2}
            actual_difficulties = {
                difficulty: difficulty_counts[domain][difficulty]
                for difficulty in ALLOWED_DIFFICULTIES
            }
            if actual_difficulties != expected_difficulties:
                warnings.append(
                    f"Domain {domain!r} difficulty balance is "
                    f"{actual_difficulties}; the design target is "
                    f"{expected_difficulties}."
                )

    counts = {
        "scenarios": len(scenarios),
        "branches": len(scenarios) * 2,
        "prefix_rows_expected": len(scenarios) * 8,
        "supervised_rows_expected": len(scenarios) * 4,
        "primary_turn4_rows_expected": len(scenarios) * 2,
        "domains": {domain: domain_counts[domain] for domain in ALLOWED_DOMAINS},
        "difficulties_by_domain": {
            domain: {
                difficulty: difficulty_counts[domain][difficulty]
                for difficulty in ALLOWED_DIFFICULTIES
            }
            for domain in ALLOWED_DOMAINS
        },
    }
    return ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        counts=counts,
        for_freeze=for_freeze,
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file and report the exact invalid line."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    scenarios: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}."
                ) from exc
            if not isinstance(value, dict):
                raise DatasetValidationError(
                    f"{path}:{line_number}: each line must be a JSON object."
                )
            scenarios.append(value)
    return scenarios


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def normalize_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize text/order and refresh deterministic audit fields."""

    normalized = [_normalize_value(copy.deepcopy(item)) for item in scenarios]
    for scenario in normalized:
        if status is not None:
            scenario["status"] = status
        if isinstance(scenario.get("audit"), dict) and all(
            isinstance(scenario.get(branch), dict)
            and isinstance(scenario[branch].get("user_turn_3"), str)
            for branch in ("benign", "suspicious")
        ):
            scenario["audit"]["turn_1_exact_match"] = True
            scenario["audit"]["turn_2_exact_match"] = True
            scenario["audit"]["turn_4_exact_match"] = True
            scenario["audit"]["assistant_stubs_exact_match"] = True
            scenario["audit"]["turn_3_word_count_ratio"] = round(
                turn3_word_count_ratio(scenario), 6
            )
    return sorted(normalized, key=lambda item: item.get("scenario_id", ""))


def canonical_dataset_bytes(scenarios: list[dict[str, Any]]) -> bytes:
    """Serialize one normalized scenario per line deterministically."""

    lines = [
        json.dumps(
            scenario,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for scenario in normalize_scenarios(scenarios)
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def compute_dataset_hash(path: str | Path) -> str:
    """Hash the exact bytes of an on-disk frozen dataset."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_scenarios_hash(scenarios: list[dict[str, Any]]) -> str:
    """Hash the deterministic representation of in-memory scenarios."""

    return hashlib.sha256(canonical_dataset_bytes(scenarios)).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def read_frozen_hash(path: str | Path) -> str:
    """Read and validate a SHA-256 digest file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen dataset hash file not found: {path}")
    digest = path.read_text(encoding="utf-8").strip().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DatasetValidationError(f"Invalid SHA-256 value in {path}.")
    return digest


def verify_frozen_dataset(
    dataset_path: str | Path,
    hash_path: str | Path,
) -> str:
    """Fail if the current dataset bytes differ from the recorded freeze hash."""

    expected = read_frozen_hash(hash_path)
    actual = compute_dataset_hash(dataset_path)
    if actual != expected:
        raise DatasetValidationError(
            "Frozen dataset hash mismatch: expected "
            f"{expected}, found {actual}. Do not continue with model execution."
        )
    return actual


def compute_length_statistics(
    scenarios: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return branch/domain word and character length audit rows."""

    rows: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda item: item["scenario_id"]):
        ratio = turn3_word_count_ratio(scenario)
        for branch in ("benign", "suspicious"):
            turn3 = scenario[branch]["user_turn_3"]
            messages = build_full_messages(scenario, branch, 4)
            full_text = plain_text_transcript(messages)
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "domain": scenario["domain"],
                    "branch": branch,
                    "turn3_word_count": _word_count(turn3),
                    "turn3_character_count": len(turn3),
                    "turn3_pair_word_count_ratio": ratio,
                    "full_t4_word_count": _word_count(full_text),
                    "full_t4_character_count": len(full_text),
                    "ratio_above_review_target": ratio > 1.25,
                }
            )
    return pd.DataFrame(rows)


def _ngrams(tokens: list[str]) -> set[str]:
    values = set(tokens)
    values.update(
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )
    return values


def find_lexical_shortcuts(
    scenarios: list[dict[str, Any]],
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """Rank unigram/bigram branch imbalances overall and within each domain."""

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, Iterable[dict[str, Any]]]] = [
        ("overall", scenarios)
    ]
    scopes.extend(
        (domain, [item for item in scenarios if item["domain"] == domain])
        for domain in ALLOWED_DOMAINS
    )

    for scope, scoped_scenarios_iter in scopes:
        scoped_scenarios = list(scoped_scenarios_iter)
        if not scoped_scenarios:
            continue
        branch_documents = {
            branch: [
                _ngrams(_words(item[branch]["user_turn_3"]))
                for item in scoped_scenarios
            ]
            for branch in ("benign", "suspicious")
        }
        vocabulary = set().union(
            *(branch_documents["benign"] + branch_documents["suspicious"])
        )
        scored: list[dict[str, Any]] = []
        n = len(scoped_scenarios)
        for term in vocabulary:
            benign_count = sum(
                term in document for document in branch_documents["benign"]
            )
            suspicious_count = sum(
                term in document for document in branch_documents["suspicious"]
            )
            if benign_count + suspicious_count < (2 if n > 1 else 1):
                continue
            benign_log_odds = math.log(
                (benign_count + 0.5) / (n - benign_count + 0.5)
            )
            suspicious_log_odds = math.log(
                (suspicious_count + 0.5) / (n - suspicious_count + 0.5)
            )
            imbalance = suspicious_log_odds - benign_log_odds
            scored.append(
                {
                    "scope": scope,
                    "term": term,
                    "ngram_size": len(term.split()),
                    "benign_document_frequency": benign_count,
                    "suspicious_document_frequency": suspicious_count,
                    "suspicious_minus_benign_frequency": (
                        suspicious_count - benign_count
                    ),
                    "log_odds_imbalance": imbalance,
                    "favours_branch": (
                        "suspicious" if imbalance > 0 else "benign"
                    ),
                    "explicit_label_term": term in EXPLICIT_LABEL_TERMS,
                }
            )
        scored.sort(
            key=lambda item: (
                -abs(item["log_odds_imbalance"]),
                item["term"],
            )
        )
        rows.extend(scored[:top_n])
    return pd.DataFrame(rows)


def _review_hash(scenario: dict[str, Any]) -> str:
    return _scenario_text_signature(scenario)


def _blind_audit_frames(
    scenarios: list[dict[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda item: item["scenario_id"]):
        for branch in ("benign", "suspicious"):
            transcript = plain_text_transcript(
                build_full_messages(scenario, branch, 3)
            )
            entries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "domain": scenario["domain"],
                    "branch": branch,
                    "expected_label": scenario[branch]["context_label_turn_3"],
                    "history_sha256": hashlib.sha256(
                        transcript.encode("utf-8")
                    ).hexdigest(),
                    "full_text_plain": transcript,
                }
            )
    random.Random(seed).shuffle(entries)

    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        blind_id = f"blind_{index:03d}"
        sheet_rows.append(
            {
                "blind_id": blind_id,
                "history_sha256": entry["history_sha256"],
                "full_text_plain": entry["full_text_plain"],
                "researcher_label": "",
                "researcher_notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "history_sha256": entry["history_sha256"],
                "scenario_id": entry["scenario_id"],
                "domain": entry["domain"],
                "branch": entry["branch"],
                "expected_label": entry["expected_label"],
            }
        )
    return pd.DataFrame(sheet_rows), pd.DataFrame(key_rows)


def _pairwise_audit_frame(
    scenarios: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for scenario in sorted(scenarios, key=lambda item: item["scenario_id"]):
        row: dict[str, Any] = {
            "scenario_id": scenario["scenario_id"],
            "scenario_content_hash": _review_hash(scenario),
        }
        row.update({column: "" for column in PAIRWISE_REVIEW_COLUMNS})
        row["reviewer"] = ""
        row["review_notes"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _assert_review_template_current(
    existing_path: Path,
    expected: pd.DataFrame,
    immutable_columns: tuple[str, ...],
) -> None:
    existing = pd.read_csv(existing_path, dtype=str, keep_default_na=False)
    if list(existing.columns) != list(expected.columns):
        raise DatasetValidationError(
            f"Existing review sheet has an unexpected schema: {existing_path}. "
            "Use --refresh-review-templates before completing the review."
        )
    if len(existing) != len(expected) or not existing[
        list(immutable_columns)
    ].equals(expected[list(immutable_columns)].astype(str)):
        raise DatasetValidationError(
            f"Existing review sheet is stale for the current draft: {existing_path}. "
            "Use --refresh-review-templates and repeat the affected review."
        )


def write_review_templates(
    scenarios: list[dict[str, Any]],
    audit_dir: str | Path,
    *,
    seed: int = 42,
    refresh: bool = False,
) -> None:
    """Create non-destructive blinded and pairwise human-review sheets."""

    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    blind_sheet, blind_key = _blind_audit_frames(scenarios, seed)
    pairwise_sheet = _pairwise_audit_frame(scenarios)

    blind_path = audit_dir / "blinded_label_audit.csv"
    pairwise_path = audit_dir / "pairwise_semantic_audit.csv"
    if blind_path.exists() and not refresh:
        _assert_review_template_current(
            blind_path,
            blind_sheet,
            ("blind_id", "history_sha256", "full_text_plain"),
        )
    else:
        _write_csv_atomic(blind_sheet, blind_path)

    if pairwise_path.exists() and not refresh:
        _assert_review_template_current(
            pairwise_path,
            pairwise_sheet,
            ("scenario_id", "scenario_content_hash"),
        )
    else:
        _write_csv_atomic(pairwise_sheet, pairwise_path)

    _write_csv_atomic(blind_key, audit_dir / "blinded_label_key.csv")


def validate_completed_human_audits(
    scenarios: list[dict[str, Any]],
    audit_dir: str | Path,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], list[str]]:
    """Verify human audit sheets match this exact draft and are complete."""

    audit_dir = Path(audit_dir)
    blind_path = audit_dir / "blinded_label_audit.csv"
    pairwise_path = audit_dir / "pairwise_semantic_audit.csv"
    errors: list[str] = []
    if not blind_path.is_file():
        errors.append(f"Missing blinded label audit: {blind_path}.")
    if not pairwise_path.is_file():
        errors.append(f"Missing pairwise semantic audit: {pairwise_path}.")
    if errors:
        return {}, errors

    expected_blind, expected_key = _blind_audit_frames(scenarios, seed)
    expected_pairwise = _pairwise_audit_frame(scenarios)
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    pairwise = pd.read_csv(pairwise_path, dtype=str, keep_default_na=False)

    try:
        _assert_review_template_current(
            blind_path,
            expected_blind,
            ("blind_id", "history_sha256", "full_text_plain"),
        )
        _assert_review_template_current(
            pairwise_path,
            expected_pairwise,
            ("scenario_id", "scenario_content_hash"),
        )
    except DatasetValidationError as exc:
        errors.append(str(exc))
        return {}, errors

    normalized_labels = blind["researcher_label"].str.strip().str.lower()
    invalid_labels = sorted(set(normalized_labels) - REVIEW_LABELS)
    if invalid_labels:
        errors.append(
            "Blinded researcher_label values must be benign_evidence, "
            f"suspicious_evidence, or ambiguous; found {invalid_labels}."
        )
    ambiguous_count = int((normalized_labels == "ambiguous").sum())
    if ambiguous_count:
        errors.append(
            f"The blinded audit contains {ambiguous_count} ambiguous histories. "
            "Revise and re-review them before freezing."
        )

    completed_pairwise = pairwise.copy()
    for column in PAIRWISE_REVIEW_COLUMNS:
        values = completed_pairwise[column].str.strip().str.lower()
        bad_rows = completed_pairwise.loc[values != "yes", "scenario_id"].tolist()
        if bad_rows:
            errors.append(
                f"Pairwise audit column {column!r} must be 'yes' for every "
                f"scenario; incomplete/rejected rows: {bad_rows}."
            )
    missing_reviewers = completed_pairwise.loc[
        completed_pairwise["reviewer"].str.strip() == "", "scenario_id"
    ].tolist()
    if missing_reviewers:
        errors.append(
            f"Pairwise audit reviewer is missing for: {missing_reviewers}."
        )

    expected_by_blind_id = expected_key.set_index("blind_id")["expected_label"]
    comparable = blind.assign(
        researcher_label_normalized=normalized_labels
    ).set_index("blind_id")
    correct = comparable["researcher_label_normalized"].eq(expected_by_blind_id)
    agreement = float(correct.mean()) if len(correct) else 0.0
    if agreement <= 0.5:
        errors.append(
            f"Blinded label agreement is {agreement:.3f}, not above chance. "
            "Review label clarity before freezing."
        )

    summary = {
        "rows": len(blind),
        "agreement": agreement,
        "ambiguous_count": ambiguous_count,
        "pairwise_rows": len(pairwise),
        "all_pairwise_checks_yes": not any(
            "Pairwise audit column" in error for error in errors
        ),
    }
    return summary, errors


def write_automatic_audits(
    scenarios: list[dict[str, Any]],
    report: ValidationReport,
    audit_dir: str | Path,
) -> None:
    """Write computed structural, length, and lexical audit artifacts."""

    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    structural = report.to_dict()
    structural["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(
        audit_dir / "structural_audit.json",
        json.dumps(structural, indent=2, sort_keys=True) + "\n",
    )
    if not report.ok:
        return
    _write_csv_atomic(
        compute_length_statistics(scenarios),
        audit_dir / "length_audit.csv",
    )
    _write_csv_atomic(
        find_lexical_shortcuts(scenarios),
        audit_dir / "lexical_audit.csv",
    )


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def freeze_dataset(
    dataset_path: str | Path,
    hash_path: str | Path,
    manifest_path: str | Path,
    *,
    config_path: str | Path | None = None,
    human_audit_dir: str | Path | None = None,
    seed: int = 42,
    allow_revision: bool = False,
    revision_note: str | None = None,
) -> dict[str, Any]:
    """Validate, normalize, freeze, hash, and record one immutable dataset."""

    dataset_path = Path(dataset_path)
    hash_path = Path(hash_path)
    manifest_path = Path(manifest_path)
    draft = normalize_scenarios(read_jsonl(dataset_path))
    draft_report = validate_dataset(draft, for_freeze=False)
    if not draft_report.ok:
        raise DatasetValidationError(
            "Draft validation failed:\n- " + "\n- ".join(draft_report.errors)
        )

    frozen = normalize_scenarios(draft, status="frozen")
    frozen_report = validate_dataset(frozen, for_freeze=True)
    if not frozen_report.ok:
        raise DatasetValidationError(
            "Freeze validation failed:\n- " + "\n- ".join(frozen_report.errors)
        )

    human_summary: dict[str, Any] | None = None
    if human_audit_dir is not None:
        human_summary, human_errors = validate_completed_human_audits(
            frozen, human_audit_dir, seed=seed
        )
        if human_errors:
            raise DatasetValidationError(
                "Human audit validation failed:\n- "
                + "\n- ".join(human_errors)
            )

    previous_hash: str | None = None
    if hash_path.exists():
        previous_hash = read_frozen_hash(hash_path)
        current_hash = compute_dataset_hash(dataset_path)
        if current_hash != previous_hash and not allow_revision:
            raise DatasetValidationError(
                "The dataset differs from its existing frozen hash. Refusing "
                "to overwrite it. Use the explicit revision option with a "
                "written reason, then rerun every downstream artifact."
            )
        if allow_revision and not (revision_note and revision_note.strip()):
            raise DatasetValidationError(
                "A non-empty revision_note is required when revising a "
                "previously frozen dataset."
            )

    frozen_bytes = canonical_dataset_bytes(frozen)
    dataset_hash = hashlib.sha256(frozen_bytes).hexdigest()
    config_hash = None
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

    root = dataset_path.resolve().parents[2]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": config_hash,
        "git_commit": _git_commit(root),
        "counts": frozen_report.counts,
        "scenario_ids": [item["scenario_id"] for item in frozen],
        "human_audit": human_summary,
        "previous_dataset_sha256": (
            previous_hash if previous_hash != dataset_hash else None
        ),
        "revision_note": revision_note.strip() if revision_note else None,
    }

    _atomic_write_bytes(dataset_path, frozen_bytes)
    _atomic_write_text(hash_path, dataset_hash + "\n")
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    verify_frozen_dataset(dataset_path, hash_path)
    return manifest

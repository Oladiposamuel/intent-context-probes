"""Shared categorical values for the frozen experiment design."""

from __future__ import annotations

from typing import Literal


ContextLabel = Literal[
    "underdetermined",
    "benign_evidence",
    "suspicious_evidence",
]
Branch = Literal["benign", "suspicious"]
Difficulty = Literal["clear", "moderate", "subtle"]

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


def binary_target(label: str) -> int | None:
    """Map the supervised evidence labels to the binary probe target."""

    if label == "underdetermined":
        return None
    if label == "benign_evidence":
        return 0
    if label == "suspicious_evidence":
        return 1
    raise ValueError(f"Unknown context label: {label!r}")

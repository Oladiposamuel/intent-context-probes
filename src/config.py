"""Configuration loading and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "models",
    "model_runtime",
    "generation",
    "probe",
    "text_baseline",
    "evaluation",
    "domains",
}

OUTPUT_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "data/audits",
    "annotations",
    "artifacts/model_metadata",
    "artifacts/activations",
    "artifacts/responses",
    "artifacts/prompted_judgements",
    "artifacts/fitted_probes",
    "artifacts/smoke_tests",
    "results",
    "figures",
    "report",
    "logs",
)


def repository_root() -> Path:
    """Return the repository root based on this module's location."""

    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration and reject incomplete core settings."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing top-level configuration keys: {missing}")

    aliases = [entry.get("alias") for entry in config["models"]]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Model aliases must be unique.")

    required_aliases = {"qwen3_4b", "qwen3_4b_saferl"}
    if not required_aliases.issubset(aliases):
        raise ValueError("Configuration must contain qwen3_4b and qwen3_4b_saferl.")

    for model in config["models"]:
        revision = model.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                f"Model {model.get('alias')!r} must pin a 40-character "
                "lowercase Hugging Face revision SHA."
            )

    if config["model_runtime"].get("batch_size") != 1:
        raise ValueError("The frozen initial protocol requires batch_size: 1.")

    return config


def get_model_spec(config: dict[str, Any], alias: str) -> dict[str, Any]:
    """Resolve one configured model by its stable experiment alias."""

    for entry in config["models"]:
        if entry.get("alias") == alias:
            return dict(entry)
    known = ", ".join(entry["alias"] for entry in config["models"])
    raise KeyError(f"Unknown model alias {alias!r}. Known aliases: {known}")


def ensure_output_directories(root: Path | None = None) -> list[Path]:
    """Create the expected local output directories."""

    root = root or repository_root()
    paths = [root / relative for relative in OUTPUT_DIRECTORIES]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths

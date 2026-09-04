"""Environment inspection and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ensure_output_directories, repository_root


REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "joblib",
    "tqdm",
    "PyYAML",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> tuple[dict[str, str | None], list[str]]:
    """Collect required package versions and missing-package errors."""

    versions: dict[str, str | None] = {}
    errors: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            errors.append(f"Missing required package: {package}")
    return versions, errors


def git_commit_sha(root: Path) -> str | None:
    """Return HEAD when the repository contains a commit."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _gpu_manifest() -> tuple[dict[str, Any], list[str]]:
    try:
        import torch
    except ImportError:
        return {"available": False}, ["PyTorch is unavailable; GPU cannot be checked."]

    if not torch.cuda.is_available():
        return {"available": False}, ["CUDA GPU is unavailable."]

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    details = {
        "available": True,
        "device_index": index,
        "name": torch.cuda.get_device_name(index),
        "memory_bytes": int(properties.total_memory),
        "memory_gib": round(properties.total_memory / 1024**3, 2),
        "cuda_version": torch.version.cuda,
    }
    errors = []
    if properties.total_memory < 15 * 1024**3:
        errors.append(
            "GPU has less than 15 GiB VRAM; Qwen3-4B float16 may not fit."
        )
    return details, errors


def _writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def build_environment_manifest(
    config_path: str | Path,
    require_gpu: bool = True,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Inspect the runtime and return manifest, errors and warnings."""

    root = repository_root()
    ensure_output_directories(root)
    errors: list[str] = []
    warnings: list[str] = []

    versions, package_errors = package_versions()
    errors.extend(package_errors)

    gpu, gpu_errors = _gpu_manifest()
    if require_gpu:
        errors.extend(gpu_errors)
    else:
        warnings.extend(gpu_errors)

    local_artifact_root = root / "artifacts"
    if not _writable_directory(local_artifact_root):
        errors.append(f"Artifact directory is not writable: {local_artifact_root}")

    persistent_value = os.environ.get("MATS_PERSISTENT_ARTIFACT_ROOT")
    persistent_root = Path(persistent_value).expanduser() if persistent_value else None
    persistent_writable = None
    if persistent_root is not None:
        persistent_writable = _writable_directory(persistent_root)
        if not persistent_writable:
            errors.append(
                f"Persistent artifact directory is not writable: {persistent_root}"
            )
    else:
        warnings.append(
            "MATS_PERSISTENT_ARTIFACT_ROOT is unset; Colab outputs are not yet "
            "protected from runtime loss."
        )

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 25 * 1024**3:
        warnings.append("Less than 25 GiB of local disk space is available.")

    dataset_path = root / "data/raw/scenarios.jsonl"
    frozen_hash_path = root / "data/FROZEN_DATASET.sha256"

    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": versions,
        "gpu": gpu,
        "git_commit_sha": git_commit_sha(root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path) if dataset_path.is_file() else None,
        "frozen_dataset_hash": (
            frozen_hash_path.read_text(encoding="utf-8").strip()
            if frozen_hash_path.is_file()
            else None
        ),
        "local_artifact_root": str(local_artifact_root),
        "persistent_artifact_root": str(persistent_root) if persistent_root else None,
        "persistent_artifact_root_writable": persistent_writable,
        "local_free_disk_bytes": free_bytes,
        "errors": errors,
        "warnings": warnings,
    }
    return manifest, errors, warnings


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write stable, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def sync_directory_to_persistent(local_directory: Path) -> Path | None:
    """Copy one artifact directory to the configured persistent root."""

    persistent_value = os.environ.get("MATS_PERSISTENT_ARTIFACT_ROOT")
    if not persistent_value:
        return None
    root = repository_root()
    relative = local_directory.resolve().relative_to((root / "artifacts").resolve())
    destination = Path(persistent_value).expanduser() / relative
    shutil.copytree(local_directory, destination, dirs_exist_ok=True)
    return destination

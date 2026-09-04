#!/usr/bin/env python3
"""Build and validate full-history/current-message prefix tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_prefixes import (  # noqa: E402
    flatten_scenarios,
    validate_prefix_dataframe,
    write_prefix_outputs,
)
from src.config import load_config  # noqa: E402
from src.data_validation import (  # noqa: E402
    DatasetValidationError,
    compute_scenarios_hash,
    normalize_scenarios,
    read_jsonl,
    validate_dataset,
    verify_frozen_dataset,
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited model/text prefix tables from scenarios."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--input")
    parser.add_argument(
        "--freeze-hash", default="data/FROZEN_DATASET.sha256"
    )
    parser.add_argument("--csv-output")
    parser.add_argument("--parquet-output")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Build preview prefixes without a frozen hash.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = _path(args.config)
        config = load_config(config_path)
        dataset_path = _path(args.input or config["project"]["dataset_path"])
        configured_parquet = _path(config["project"]["processed_path"])
        parquet_path = _path(args.parquet_output) if args.parquet_output else configured_parquet
        csv_path = (
            _path(args.csv_output)
            if args.csv_output
            else parquet_path.with_suffix(".csv")
        )

        scenarios = normalize_scenarios(read_jsonl(dataset_path))
        if args.allow_draft:
            report = validate_dataset(scenarios, for_freeze=False)
            dataset_hash = compute_scenarios_hash(scenarios)
        else:
            dataset_hash = verify_frozen_dataset(
                dataset_path, _path(args.freeze_hash)
            )
            report = validate_dataset(scenarios, for_freeze=True)
        if not report.ok:
            raise DatasetValidationError(
                "Dataset validation failed:\n- " + "\n- ".join(report.errors)
            )

        frame = flatten_scenarios(scenarios, dataset_hash)
        prefix_errors = validate_prefix_dataframe(frame, len(scenarios))
        if prefix_errors:
            raise DatasetValidationError(
                "Prefix validation failed:\n- " + "\n- ".join(prefix_errors)
            )
        write_prefix_outputs(frame, parquet_path, csv_path)
        print(f"PREFIX BUILD PASSED: {len(frame)} rows")
        print(f"Dataset SHA-256: {dataset_hash}")
        print(f"CSV: {csv_path}")
        print(f"Parquet: {parquet_path}")
        return 0
    except (DatasetValidationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

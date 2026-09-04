#!/usr/bin/env python3
"""Validate draft scenarios, prepare audits, and explicitly freeze the dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import (  # noqa: E402
    DatasetValidationError,
    freeze_dataset,
    normalize_scenarios,
    read_jsonl,
    validate_dataset,
    verify_frozen_dataset,
    write_automatic_audits,
    write_review_templates,
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a partial draft or explicitly freeze the reviewed "
            "32-scenario dataset."
        )
    )
    parser.add_argument("--input", default="data/raw/scenarios.jsonl")
    parser.add_argument("--audit-dir", default="data/audits")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Explicitly normalize and freeze the reviewed complete dataset.",
    )
    parser.add_argument(
        "--freeze-hash", default="data/FROZEN_DATASET.sha256"
    )
    parser.add_argument(
        "--manifest", default="data/audits/freeze_manifest.json"
    )
    parser.add_argument(
        "--refresh-review-templates",
        action="store_true",
        help=(
            "Replace human-review sheets for the current draft. This removes "
            "any labels already entered into those sheets."
        ),
    )
    parser.add_argument(
        "--allow-revision",
        action="store_true",
        help="Explicitly replace an existing frozen dataset after correction.",
    )
    parser.add_argument(
        "--revision-note",
        help="Required written reason when --allow-revision is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _print_report(report) -> None:
    print(json.dumps(report.counts, indent=2, sort_keys=True))
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    dataset_path = _path(args.input)
    audit_dir = _path(args.audit_dir)
    hash_path = _path(args.freeze_hash)
    manifest_path = _path(args.manifest)
    config_path = _path(args.config)

    try:
        if args.freeze and args.refresh_review_templates:
            raise DatasetValidationError(
                "Do not combine --freeze with --refresh-review-templates; "
                "refresh first, complete the reviews, then freeze separately."
            )
        if args.allow_revision and not args.freeze:
            raise DatasetValidationError(
                "--allow-revision is valid only together with --freeze."
            )
        if args.revision_note and not args.allow_revision:
            raise DatasetValidationError(
                "--revision-note requires --allow-revision."
            )
        raw_scenarios = read_jsonl(dataset_path)
        scenarios = normalize_scenarios(raw_scenarios)
        report = validate_dataset(scenarios, for_freeze=False)
        write_automatic_audits(scenarios, report, audit_dir)
        _print_report(report)
        if not report.ok:
            return 1

        write_review_templates(
            scenarios,
            audit_dir,
            seed=args.seed,
            refresh=args.refresh_review_templates,
        )

        if not args.freeze:
            if hash_path.exists():
                digest = verify_frozen_dataset(dataset_path, hash_path)
                print(f"FROZEN DATASET VERIFIED: {digest}")
            else:
                print("DRAFT VALIDATION PASSED; dataset was not frozen.")
                print(
                    "Complete the pairwise and blinded audit sheets before "
                    "running this command with --freeze."
                )
            return 0

        manifest = freeze_dataset(
            dataset_path,
            hash_path,
            manifest_path,
            config_path=config_path,
            human_audit_dir=audit_dir,
            seed=args.seed,
            allow_revision=args.allow_revision,
            revision_note=args.revision_note,
        )
        frozen_scenarios = read_jsonl(dataset_path)
        frozen_report = validate_dataset(frozen_scenarios, for_freeze=True)
        write_automatic_audits(frozen_scenarios, frozen_report, audit_dir)
        print(f"DATASET FROZEN: {manifest['dataset_sha256']}")
        print(f"Freeze manifest: {manifest_path}")
        return 0
    except (DatasetValidationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

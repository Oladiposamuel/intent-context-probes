from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import (
    DatasetValidationError,
    PAIRWISE_REVIEW_COLUMNS,
    canonical_dataset_bytes,
    compute_scenarios_hash,
    freeze_dataset,
    read_jsonl,
    validate_dataset,
    verify_frozen_dataset,
    write_automatic_audits,
    write_review_templates,
)
from tests.fixtures import make_complete_dataset, make_scenario


def _write_scenarios(path: Path, scenarios) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_dataset_bytes(scenarios))


def _complete_reviews(audit_dir: Path) -> None:
    blind_path = audit_dir / "blinded_label_audit.csv"
    key_path = audit_dir / "blinded_label_key.csv"
    blind = pd.read_csv(blind_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    expected = key.set_index("blind_id")["expected_label"]
    blind["researcher_label"] = blind["blind_id"].map(expected)
    blind.to_csv(blind_path, index=False, lineterminator="\n")

    pairwise_path = audit_dir / "pairwise_semantic_audit.csv"
    pairwise = pd.read_csv(pairwise_path, dtype=str, keep_default_na=False)
    for column in PAIRWISE_REVIEW_COLUMNS:
        pairwise[column] = "yes"
    pairwise["reviewer"] = "Samuel Oladipo"
    pairwise.to_csv(pairwise_path, index=False, lineterminator="\n")


class DataValidationTests(unittest.TestCase):
    def test_partial_draft_can_be_validated_without_freezing(self):
        report = validate_dataset([make_scenario()], for_freeze=False)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.counts["scenarios"], 1)

    def test_freeze_requires_exact_counts_per_domain(self):
        scenario = make_scenario(status="frozen", manual_label_confirmed=True)
        report = validate_dataset([scenario], for_freeze=True)
        self.assertFalse(report.ok)
        self.assertIn("exactly 32 scenarios", "\n".join(report.errors))

    def test_complete_balanced_dataset_is_freeze_valid(self):
        report = validate_dataset(make_complete_dataset(), for_freeze=True)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(
            report.counts["domains"],
            {
                "cybersecurity": 8,
                "biosecurity_public_health": 8,
                "chemical_safety": 8,
                "autonomous_physical_security": 8,
            },
        )

    def test_duplicate_ids_and_conversation_text_are_rejected(self):
        first = make_scenario()
        second = copy.deepcopy(first)
        report = validate_dataset([first, second])
        errors = "\n".join(report.errors)
        self.assertIn("Duplicate scenario IDs", errors)
        self.assertIn("duplicates all conversation text", errors)

    def test_invalid_label_transition_is_rejected(self):
        scenario = make_scenario()
        scenario["suspicious"]["context_label_turn_4"] = "benign_evidence"
        report = validate_dataset([scenario])
        self.assertIn(
            "suspicious.context_label_turn_4 must be 'suspicious_evidence'",
            "\n".join(report.errors),
        )

    def test_structural_audit_is_written_even_for_invalid_schema(self):
        scenario = make_scenario()
        del scenario["shared"]["user_turn_2"]
        report = validate_dataset([scenario])
        with tempfile.TemporaryDirectory() as temporary:
            audit_dir = Path(temporary)
            write_automatic_audits([scenario], report, audit_dir)
            payload = json.loads(
                (audit_dir / "structural_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(payload["ok"])
            self.assertFalse((audit_dir / "length_audit.csv").exists())

    def test_canonical_hash_is_independent_of_input_order(self):
        scenarios = make_complete_dataset()[:2]
        self.assertEqual(
            compute_scenarios_hash(scenarios),
            compute_scenarios_hash(list(reversed(scenarios))),
        )

    def test_freeze_and_hash_verification_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "data/raw/scenarios.jsonl"
            hash_path = root / "data/FROZEN_DATASET.sha256"
            audit_dir = root / "data/audits"
            manifest_path = audit_dir / "freeze_manifest.json"
            config_path = root / "configs/experiment.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("project: test\n", encoding="utf-8")

            _write_scenarios(dataset_path, make_complete_dataset())
            scenarios = read_jsonl(dataset_path)
            write_review_templates(scenarios, audit_dir)
            _complete_reviews(audit_dir)
            manifest = freeze_dataset(
                dataset_path,
                hash_path,
                manifest_path,
                config_path=config_path,
                human_audit_dir=audit_dir,
            )
            self.assertEqual(
                verify_frozen_dataset(dataset_path, hash_path),
                manifest["dataset_sha256"],
            )
            self.assertEqual(len(read_jsonl(dataset_path)), 32)

            dataset_path.write_text(
                dataset_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError):
                verify_frozen_dataset(dataset_path, hash_path)

    def test_freeze_requires_completed_human_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "data/raw/scenarios.jsonl"
            _write_scenarios(dataset_path, make_complete_dataset())
            audit_dir = root / "data/audits"
            write_review_templates(read_jsonl(dataset_path), audit_dir)
            with self.assertRaises(DatasetValidationError):
                freeze_dataset(
                    dataset_path,
                    root / "data/FROZEN_DATASET.sha256",
                    audit_dir / "freeze_manifest.json",
                    human_audit_dir=audit_dir,
                )


if __name__ == "__main__":
    unittest.main()

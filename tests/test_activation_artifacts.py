from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_prefixes import flatten_scenarios
from src.extract_activations import (
    load_activation_checkpoint,
    run_activation_condition,
    save_activation_checkpoint,
    validate_activation_artifact,
    verify_smoke_test_artifact,
)
from src.model_loading import ModelBundle
from tests.fixtures import make_scenario


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        body = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return body + "|assistant:"


class ActivationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.layers = [1, 2]
        self.expected_ids = ["example_a", "example_b"]
        self.metadata = {
            "model_id": "org/model",
            "model_revision": "a" * 40,
            "context_mode": "full_history",
            "dataset_hash": "b" * 64,
            "hidden_size": 3,
        }
        self.records = {
            "example_a": {
                "example_id": "example_a",
                "input_hash": "hash_a",
                "input_token_count": 10,
                "readout_token_id": 7,
                "vectors": {
                    1: np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    2: np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
                },
            },
            "example_b": {
                "example_id": "example_b",
                "input_hash": "hash_b",
                "input_token_count": 11,
                "readout_token_id": 8,
                "vectors": {
                    1: np.asarray([7.0, 8.0, 9.0], dtype=np.float32),
                    2: np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                },
            },
        }

    def test_checkpoint_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_history.npz"
            save_activation_checkpoint(
                self.records,
                self.expected_ids,
                self.layers,
                path,
                self.metadata,
            )
            validate_activation_artifact(
                path,
                self.expected_ids,
                self.layers,
                hidden_size=3,
            )
            loaded = load_activation_checkpoint(path, self.metadata)
            self.assertEqual(list(loaded), self.expected_ids)
            np.testing.assert_array_equal(
                loaded["example_b"]["vectors"][2],
                self.records["example_b"]["vectors"][2],
            )

    def test_checkpoint_refuses_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_history.npz"
            save_activation_checkpoint(
                self.records,
                self.expected_ids,
                self.layers,
                path,
                self.metadata,
            )
            changed = {**self.metadata, "model_revision": "c" * 40}
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                load_activation_checkpoint(path, changed)

    def test_checkpoint_preserves_requested_prefix_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_history.npz"
            reverse_order = list(reversed(self.expected_ids))
            save_activation_checkpoint(
                self.records,
                reverse_order,
                self.layers,
                path,
                self.metadata,
            )
            with np.load(path, allow_pickle=False) as artifact:
                self.assertEqual(
                    artifact["example_ids"].astype(str).tolist(), reverse_order
                )

    def test_bulk_run_requires_matching_passed_smoke_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts/smoke_tests/model/smoke_test.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "model": {
                            "alias": "model",
                            "model_id": "org/model",
                            "revision": "a" * 40,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_smoke_test_artifact(root, "model", "org/model", "a" * 40),
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "pinned revision"):
                verify_smoke_test_artifact(root, "model", "org/model", "b" * 40)

    def test_condition_deduplicates_paired_inputs_and_writes_all_rows(self):
        frame = flatten_scenarios([make_scenario()], "b" * 64)
        config = {
            "domains": ["cybersecurity"],
            "model_runtime": {
                "readout": "assistant_decision_last_input_position",
                "enable_thinking": False,
                "add_generation_prompt": True,
                "max_length": 2048,
            },
        }
        bundle = ModelBundle(
            alias="model",
            model_id="org/model",
            revision="a" * 40,
            tokenizer=FakeTokenizer(),
            model=object(),
        )
        calls = []

        def fake_extract(model, tokenizer, rendered, layers, max_length):
            calls.append(rendered)
            base = float(len(calls))
            return (
                {
                    layer: np.asarray([base, base + 1, base + 2], dtype=np.float32)
                    for layer in layers
                },
                {"original_token_count": len(rendered), "final_token_id": 99},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("src.extract_activations.repository_root", return_value=root),
                patch("src.extract_activations.sync_directory_to_persistent"),
                patch(
                    "src.extract_activations.extract_selected_activations",
                    side_effect=fake_extract,
                ),
            ):
                full_path = run_activation_condition(
                    bundle,
                    frame,
                    "full_history",
                    config,
                    {"hidden_size": 3},
                    [1, 2],
                )
                self.assertEqual(len(calls), 6)
                self.assertTrue(full_path.is_file())
                with np.load(full_path, allow_pickle=False) as artifact:
                    self.assertEqual(artifact["X"].shape, (8, 2, 3))

                calls.clear()
                current_path = run_activation_condition(
                    bundle,
                    frame,
                    "current_message",
                    config,
                    {"hidden_size": 3},
                    [1, 2],
                )
                self.assertEqual(len(calls), 5)
                with np.load(current_path, allow_pickle=False) as artifact:
                    self.assertEqual(artifact["X"].shape, (8, 2, 3))


if __name__ == "__main__":
    unittest.main()

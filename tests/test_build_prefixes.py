from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_prefixes import (
    build_full_messages,
    flatten_scenarios,
    get_current_user_message,
    validate_prefix_dataframe,
)
from tests.fixtures import make_scenario


class PrefixBuilderTests(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario()
        self.frame = flatten_scenarios([self.scenario], "a" * 64)

    def test_one_scenario_produces_eight_rows(self):
        self.assertEqual(len(self.frame), 8)
        self.assertEqual(self.frame["example_id"].nunique(), 8)
        self.assertEqual(len(self.frame[self.frame["is_primary_t4"]]), 2)
        self.assertEqual(
            len(self.frame[self.frame["turn_index"].isin([3, 4])]), 4
        )

    def test_turn4_current_messages_match_but_histories_differ_at_turn3(self):
        benign = build_full_messages(self.scenario, "benign", 4)
        suspicious = build_full_messages(self.scenario, "suspicious", 4)
        differing = [
            index
            for index, pair in enumerate(zip(benign, suspicious))
            if pair[0] != pair[1]
        ]
        self.assertEqual(differing, [4])
        self.assertEqual(
            get_current_user_message(self.scenario, "benign", 4),
            get_current_user_message(self.scenario, "suspicious", 4),
        )

    def test_targets_use_context_while_turn4_local_label_is_underdetermined(self):
        turn4 = self.frame[self.frame["turn_index"] == 4].set_index("branch")
        self.assertEqual(turn4.loc["benign", "binary_target"], 0)
        self.assertEqual(turn4.loc["suspicious", "binary_target"], 1)
        self.assertEqual(set(turn4["local_label"]), {"underdetermined"})

    def test_full_prefix_json_is_valid_and_role_alternating(self):
        for payload in self.frame["messages_full_json"]:
            messages = json.loads(payload)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[-1]["role"], "user")

    def test_prefix_validator_detects_changed_turn4_current_message(self):
        changed = self.frame.copy()
        mask = (
            (changed["branch"] == "suspicious")
            & (changed["turn_index"] == 4)
        )
        changed.loc[mask, "current_user_message"] = "Different message"
        errors = "\n".join(validate_prefix_dataframe(changed, 1))
        self.assertIn("current messages must be identical", errors)

    def test_valid_prefix_frame_passes_all_invariants(self):
        self.assertEqual(validate_prefix_dataframe(self.frame, 1), [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas import validate_required_fields
from tests.fixtures import make_scenario


class ScenarioSchemaTests(unittest.TestCase):
    def test_complete_scenario_is_schema_valid(self):
        self.assertEqual(validate_required_fields(make_scenario()), [])

    def test_missing_and_unknown_fields_are_rejected(self):
        scenario = make_scenario()
        del scenario["subtopic"]
        scenario["topic_typo"] = "value"
        errors = "\n".join(validate_required_fields(scenario))
        self.assertIn("missing required fields", errors)
        self.assertIn("unknown fields", errors)

    def test_adapted_source_requires_reference(self):
        scenario = make_scenario()
        scenario["source_type"] = "adapted"
        errors = "\n".join(validate_required_fields(scenario))
        self.assertIn("source_reference is required", errors)

    def test_unknown_nested_branch_field_is_rejected(self):
        scenario = copy.deepcopy(make_scenario())
        scenario["benign"]["hidden_label"] = "benign"
        errors = "\n".join(validate_required_fields(scenario))
        self.assertIn("scenario.benign contains unknown fields", errors)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompted_judge import build_judge_prompt, parse_judge_json


class PromptedJudgeTests(unittest.TestCase):
    def test_build_prompt_uses_observable_evidence_language(self):
        prompt = build_judge_prompt("USER: Example")
        self.assertIn("observable evidence", prompt[0]["content"])
        self.assertIn("USER: Example", prompt[0]["content"])

    def test_parse_valid_json_inside_wrapper_text(self):
        parsed = parse_judge_json(
            'answer: {"score": 72, "label": "suspicious_evidence", '
            '"brief_reason": "The request seeks evasion."}'
        )
        self.assertEqual(parsed["score"], 72)

    def test_parse_rejects_invalid_score(self):
        with self.assertRaises(ValueError):
            parse_judge_json(
                '{"score": 101, "label": "underdetermined", "brief_reason": "x"}'
            )


if __name__ == "__main__":
    unittest.main()

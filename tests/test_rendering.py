from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rendering import hash_rendered_input, render_chat, validate_messages
from src.smoke_test import current_turn4_messages, full_turn4_messages


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.received = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        body = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return body + "|assistant:"


class RenderingTests(unittest.TestCase):
    def test_full_histories_differ_only_at_turn_three_content(self):
        benign = full_turn4_messages("benign")
        suspicious = full_turn4_messages("suspicious")
        self.assertEqual(len(benign), len(suspicious))
        differing = [i for i, pair in enumerate(zip(benign, suspicious)) if pair[0] != pair[1]]
        self.assertEqual(differing, [4])
        self.assertEqual(benign[-1], suspicious[-1])

    def test_current_turn_four_is_single_identical_user_message(self):
        messages = current_turn4_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    def test_rendering_requests_nonthinking_generation_marker(self):
        tokenizer = FakeTokenizer()
        rendered = render_chat(tokenizer, current_turn4_messages())
        self.assertTrue(rendered.endswith("|assistant:"))
        self.assertEqual(
            tokenizer.received,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_input_hash_includes_metadata(self):
        first = hash_rendered_input("same", {"model": "a"})
        second = hash_rendered_input("same", {"model": "b"})
        self.assertNotEqual(first, second)

    def test_history_must_end_in_user_message(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "assistant", "content": "No."}])


if __name__ == "__main__":
    unittest.main()

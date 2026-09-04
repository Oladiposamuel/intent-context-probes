from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schemas import binary_target


class SchemaTests(unittest.TestCase):
    def test_binary_target_mapping(self):
        self.assertIsNone(binary_target("underdetermined"))
        self.assertEqual(binary_target("benign_evidence"), 0)
        self.assertEqual(binary_target("suspicious_evidence"), 1)

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            binary_target("malicious")


if __name__ == "__main__":
    unittest.main()

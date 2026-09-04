from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract_activations import selected_layer_indices


class LayerSelectionTests(unittest.TestCase):
    def test_expected_layers_for_36_layer_model(self):
        self.assertEqual(
            selected_layer_indices(36, [0.25, 0.50, 0.75, 1.00]),
            [9, 18, 27, 36],
        )

    def test_duplicate_rounded_layers_are_removed(self):
        self.assertEqual(selected_layer_indices(2, [0.25, 0.50, 1.00]), [1, 2])

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            selected_layer_indices(36, [0.0])


if __name__ == "__main__":
    unittest.main()

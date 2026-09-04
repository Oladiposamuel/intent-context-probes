from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_model_spec, load_config


class ConfigTests(unittest.TestCase):
    def test_experiment_config_loads(self):
        config = load_config(ROOT / "configs/experiment.yaml")
        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["model_runtime"]["batch_size"], 1)

    def test_both_subject_models_exist(self):
        config = load_config(ROOT / "configs/experiment.yaml")
        self.assertEqual(
            get_model_spec(config, "qwen3_4b")["model_id"],
            "Qwen/Qwen3-4B",
        )
        self.assertEqual(
            get_model_spec(config, "qwen3_4b_saferl")["model_id"],
            "Qwen/Qwen3-4B-SafeRL",
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run a model smoke test or frozen-prefix activation extraction."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import get_model_spec, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--extract-activations", action="store_true")
    parser.add_argument("--generate-turn4-responses", action="store_true")
    parser.add_argument("--run-prompted-judge", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import os

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path

    config = load_config(config_path)
    get_model_spec(config, args.model)

    output_flags = args.generate_turn4_responses or args.run_prompted_judge
    if (args.smoke_test or args.extract_activations) and output_flags:
        print(
            "Smoke/extraction and behavioural outputs must be separate runs.",
            file=sys.stderr,
        )
        return 2
    if args.smoke_test and args.extract_activations:
        print(
            "Pass exactly one of --smoke-test or --extract-activations so the "
            "smoke artifact can be inspected before the bulk run.",
            file=sys.stderr,
        )
        return 2
    if not args.smoke_test and not args.extract_activations and not output_flags:
        print(
            "Nothing selected. Choose a model operation.",
            file=sys.stderr,
        )
        return 2

    set_seed(int(config["project"]["seed"]))
    if args.smoke_test:
        from src.smoke_test import run_smoke_test

        run_smoke_test(config, args.model)
    elif args.extract_activations:
        from src.extract_activations import run_bulk_activation_extraction

        run_bulk_activation_extraction(config, args.model)
    else:
        from src.model_outputs import run_model_outputs

        run_model_outputs(
            config,
            args.model,
            responses=args.generate_turn4_responses,
            judge=args.run_prompted_judge,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load one pinned model once and run selected behavioural outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_model_spec, repository_root
from .extract_activations import verify_smoke_test_artifact
from .generate_responses import generate_turn4_responses
from .model_loading import (
    ModelBundle,
    load_model,
    load_tokenizer,
    release_model,
    resolve_model_revision,
)
from .prompted_judge import run_prompted_judgements


def run_model_outputs(
    config: dict, model_alias: str, responses: bool, judge: bool
) -> list[Path]:
    root = repository_root()
    prefix_path = root / config["project"]["processed_path"]
    frame = pd.read_parquet(prefix_path)
    frozen_hash = (root / "data/FROZEN_DATASET.sha256").read_text().strip()
    if frame["dataset_hash"].drop_duplicates().tolist() != [frozen_hash]:
        raise RuntimeError("Prefix table does not match the frozen dataset hash.")
    spec = get_model_spec(config, model_alias)
    verify_smoke_test_artifact(root, model_alias, spec["model_id"], spec["revision"])
    revision = resolve_model_revision(spec["model_id"], spec["revision"])
    tokenizer = load_tokenizer(spec["model_id"], revision)
    runtime = config["model_runtime"]
    model = load_model(
        spec["model_id"], revision, runtime["dtype"], runtime["device_map"]
    )
    bundle = ModelBundle(model_alias, spec["model_id"], revision, tokenizer, model)
    outputs = []
    try:
        if responses:
            outputs.append(generate_turn4_responses(bundle, frame, config))
        if judge:
            outputs.extend(run_prompted_judgements(bundle, frame, config))
        return outputs
    finally:
        release_model(bundle)

"""Assistant-decision-position activation extraction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_prefixes import validate_prefix_dataframe
from .config import ensure_output_directories, get_model_spec, repository_root
from .environment import sha256_file, sync_directory_to_persistent
from .model_loading import (
    ModelBundle,
    collect_model_metadata,
    load_model,
    load_tokenizer,
    model_input_device,
    release_model,
    resolve_model_revision,
)
from .rendering import (
    hash_rendered_input,
    inspect_readout_token,
    render_chat,
)

CONTEXT_MODES = ("full_history", "current_message")


def selected_layer_indices(
    num_hidden_layers: int,
    fractions: Iterable[float],
) -> list[int]:
    """Convert relative depth fractions to unique transformer-layer outputs."""

    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive.")
    selected = set()
    for fraction in fractions:
        if not 0 < float(fraction) <= 1:
            raise ValueError(f"Layer fraction must be in (0, 1], got {fraction}.")
        selected.add(max(1, round(num_hidden_layers * float(fraction))))
    return sorted(selected)


def verify_smoke_test_artifact(
    root: str | Path,
    model_alias: str,
    model_id: str,
    revision: str,
) -> Path:
    """Require a passed smoke test for the exact checkpoint before bulk work."""

    path = Path(root) / "artifacts/smoke_tests" / model_alias / "smoke_test.json"
    if not path.is_file():
        raise RuntimeError(
            f"Missing smoke-test artifact for {model_alias}: {path}. "
            "Run --smoke-test and inspect it before bulk extraction."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    if payload.get("status") != "passed":
        raise RuntimeError(f"Smoke test did not pass for {model_alias}: {path}")
    if model.get("alias") != model_alias or model.get("model_id") != model_id:
        raise RuntimeError(
            f"Smoke-test model identity does not match {model_alias}/{model_id}."
        )
    if model.get("revision") != revision:
        raise RuntimeError(
            f"Smoke-test revision {model.get('revision')!r} does not match "
            f"the pinned revision {revision!r}."
        )
    return path


def extract_selected_activations(
    model,
    tokenizer,
    rendered_text: str,
    selected_layers: list[int],
    max_length: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Extract only selected final-position vectors and move them to CPU."""

    import torch

    untruncated = tokenizer(
        rendered_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    original_length = int(untruncated["input_ids"].shape[-1])
    if original_length > max_length:
        raise ValueError(
            f"Input has {original_length} tokens and would exceed "
            f"max_length={max_length}."
        )

    inputs = {
        key: value.to(model_input_device(model)) for key, value in untruncated.items()
    }
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model returned no hidden states.")
    expected_count = int(model.config.num_hidden_layers) + 1
    if len(hidden_states) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} hidden-state tensors, got {len(hidden_states)}."
        )

    vectors: dict[int, np.ndarray] = {}
    for layer in selected_layers:
        if not 1 <= layer <= model.config.num_hidden_layers:
            raise ValueError(f"Selected layer {layer} is outside model depth.")
        vector = hidden_states[layer][0, -1, :].float().cpu().numpy()
        if vector.shape != (int(model.config.hidden_size),):
            raise RuntimeError(
                f"Layer {layer} vector has unexpected shape {vector.shape}."
            )
        if not np.isfinite(vector).all():
            raise RuntimeError(f"Layer {layer} vector contains NaN or infinity.")
        vectors[layer] = vector

    token_info = inspect_readout_token(tokenizer, inputs["input_ids"])
    metadata = {
        **token_info,
        "original_token_count": original_length,
        "truncated": False,
        "hidden_state_tensor_count": len(hidden_states),
        "selected_layers": selected_layers,
        "hidden_size": int(model.config.hidden_size),
    }
    del outputs, hidden_states
    return vectors, metadata


def _messages_for_row(row: pd.Series, context_mode: str) -> list[dict[str, str]]:
    if context_mode == "full_history":
        messages = json.loads(row["messages_full_json"])
        if not isinstance(messages, list):
            raise ValueError(f"{row['example_id']}: messages_full_json is not a list.")
        return messages
    if context_mode == "current_message":
        return [{"role": "user", "content": row["current_user_message"]}]
    raise ValueError(
        f"Unknown context_mode {context_mode!r}; expected one of {CONTEXT_MODES}."
    )


def _input_metadata(
    bundle: ModelBundle,
    context_mode: str,
    selected_layers: list[int],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Return every setting that defines one activation input."""

    return {
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "context_mode": context_mode,
        "readout": runtime["readout"],
        "enable_thinking": runtime["enable_thinking"],
        "add_generation_prompt": runtime["add_generation_prompt"],
        "max_length": int(runtime["max_length"]),
        "selected_layers": selected_layers,
    }


def _ordered_records(
    records: dict[str, dict[str, Any]],
    expected_order: list[str],
) -> list[dict[str, Any]]:
    return [
        records[example_id] for example_id in expected_order if example_id in records
    ]


def save_activation_checkpoint(
    records: dict[str, dict[str, Any]],
    expected_order: list[str],
    selected_layers: list[int],
    destination: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically save selected activation rows and reproducibility metadata."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_records(records, expected_order)
    if not ordered:
        raise ValueError("Cannot save an empty activation checkpoint.")

    example_ids = np.asarray([record["example_id"] for record in ordered])
    input_hashes = np.asarray([record["input_hash"] for record in ordered])
    input_token_counts = np.asarray(
        [record["input_token_count"] for record in ordered], dtype=np.int32
    )
    readout_token_ids = np.asarray(
        [record["readout_token_id"] for record in ordered], dtype=np.int64
    )
    x = np.stack(
        [
            np.stack(
                [record["vectors"][layer] for layer in selected_layers],
                axis=0,
            )
            for record in ordered
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            example_ids=example_ids,
            layers=np.asarray(selected_layers, dtype=np.int16),
            X=x,
            input_hashes=input_hashes,
            input_token_counts=input_token_counts,
            readout_token_ids=readout_token_ids,
        )
    temporary.replace(destination)

    metadata_path = destination.with_suffix(".json")
    metadata_payload = {
        **metadata,
        "row_count": len(ordered),
        "expected_row_count": len(expected_order),
        "complete": len(ordered) == len(expected_order),
        "example_ids_sha256": hashlib.sha256(
            "\n".join(example_ids.tolist()).encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": sha256_file(destination),
    }
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return destination, metadata_path


def load_activation_checkpoint(
    destination: str | Path,
    expected_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load a compatible partial checkpoint for safe resumption."""

    destination = Path(destination)
    metadata_path = destination.with_suffix(".json")
    if not destination.exists() and not metadata_path.exists():
        return {}
    if not destination.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"Incomplete checkpoint pair at {destination}; remove or repair it."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Checkpoint metadata mismatch for {key}: "
                f"expected {value!r}, found {metadata.get(key)!r}."
            )
    if metadata.get("artifact_sha256") != sha256_file(destination):
        raise RuntimeError(f"Activation checkpoint hash mismatch: {destination}")

    with np.load(destination, allow_pickle=False) as artifact:
        required = {
            "example_ids",
            "layers",
            "X",
            "input_hashes",
            "input_token_counts",
            "readout_token_ids",
        }
        if set(artifact.files) != required:
            raise RuntimeError(
                f"Activation checkpoint keys are invalid: {sorted(artifact.files)}"
            )
        example_ids = artifact["example_ids"].astype(str)
        layers = artifact["layers"].astype(int).tolist()
        x = artifact["X"].astype(np.float32, copy=False)
        input_hashes = artifact["input_hashes"].astype(str)
        token_counts = artifact["input_token_counts"].astype(int)
        token_ids = artifact["readout_token_ids"].astype(int)

    if x.ndim != 3 or x.shape[:2] != (len(example_ids), len(layers)):
        raise RuntimeError(f"Activation checkpoint has invalid X shape: {x.shape}")
    expected_layers = expected_metadata.get("selected_layers")
    if expected_layers is not None and layers != expected_layers:
        raise RuntimeError(
            f"Checkpoint layers {layers} do not match expected {expected_layers}."
        )
    expected_hidden_size = expected_metadata.get("hidden_size")
    if expected_hidden_size is not None and x.shape[2] != expected_hidden_size:
        raise RuntimeError(
            f"Checkpoint hidden size {x.shape[2]} does not match "
            f"expected {expected_hidden_size}."
        )
    row_count = len(example_ids)
    if not all(
        len(values) == row_count for values in (input_hashes, token_counts, token_ids)
    ):
        raise RuntimeError("Activation checkpoint arrays have inconsistent lengths.")
    if metadata.get("row_count") != row_count:
        raise RuntimeError("Activation checkpoint row count disagrees with metadata.")
    if len(set(example_ids.tolist())) != len(example_ids):
        raise RuntimeError("Activation checkpoint contains duplicate example IDs.")
    if not np.isfinite(x).all():
        raise RuntimeError("Activation checkpoint contains NaN or infinity.")

    records: dict[str, dict[str, Any]] = {}
    for index, example_id in enumerate(example_ids):
        records[example_id] = {
            "example_id": example_id,
            "input_hash": input_hashes[index],
            "input_token_count": int(token_counts[index]),
            "readout_token_id": int(token_ids[index]),
            "vectors": {
                layer: x[index, layer_index] for layer_index, layer in enumerate(layers)
            },
        }
    return records


def validate_activation_artifact(
    destination: str | Path,
    expected_ids: list[str],
    selected_layers: list[int],
    hidden_size: int,
) -> None:
    """Fail loudly if a completed activation file violates its schema."""

    destination = Path(destination)
    metadata_path = destination.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise RuntimeError("Activation metadata does not mark the artifact complete.")
    if metadata.get("artifact_sha256") != sha256_file(destination):
        raise RuntimeError(
            "Completed activation artifact hash does not match metadata."
        )
    with np.load(destination, allow_pickle=False) as artifact:
        example_ids = artifact["example_ids"].astype(str).tolist()
        layers = artifact["layers"].astype(int).tolist()
        x = artifact["X"]
        hashes = artifact["input_hashes"].astype(str)
        token_counts = artifact["input_token_counts"]
        token_ids = artifact["readout_token_ids"]

    if example_ids != expected_ids:
        raise RuntimeError(
            "Activation example IDs do not match the prefix table order."
        )
    if layers != selected_layers:
        raise RuntimeError(f"Expected layers {selected_layers}; found {layers}.")
    expected_shape = (len(expected_ids), len(selected_layers), hidden_size)
    if x.shape != expected_shape:
        raise RuntimeError(
            f"Expected activation shape {expected_shape}; found {x.shape}."
        )
    if not np.isfinite(x).all():
        raise RuntimeError("Completed activation artifact contains NaN or infinity.")
    if len(hashes) != len(expected_ids) or any(not value for value in hashes):
        raise RuntimeError("Activation artifact contains missing input hashes.")
    if len(token_counts) != len(expected_ids) or np.any(token_counts <= 0):
        raise RuntimeError("Activation artifact contains invalid token counts.")
    if len(token_ids) != len(expected_ids):
        raise RuntimeError("Activation artifact contains invalid readout token IDs.")


def _assert_pair_invariants(
    frame: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    context_mode: str,
) -> None:
    """Verify exact hashes/vectors for prefixes that must be shared."""

    equal_turns = {1, 2} if context_mode == "full_history" else {1, 2, 4}
    for _, pair in frame[frame["turn_index"].isin(equal_turns)].groupby(
        "pair_turn_id", sort=True
    ):
        if len(pair) != 2:
            raise RuntimeError("Each checked pair_turn_id must contain two rows.")
        first_id, second_id = pair["example_id"].tolist()
        first = records[first_id]
        second = records[second_id]
        if first["input_hash"] != second["input_hash"]:
            raise RuntimeError(
                f"Shared pair {pair.iloc[0]['pair_turn_id']} has different "
                "input hashes."
            )
        for layer in first["vectors"]:
            if not np.array_equal(first["vectors"][layer], second["vectors"][layer]):
                raise RuntimeError(
                    f"Shared pair {pair.iloc[0]['pair_turn_id']} has different vectors."
                )


def run_activation_condition(
    bundle: ModelBundle,
    prefix_frame: pd.DataFrame,
    context_mode: str,
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    selected_layers: list[int],
) -> Path:
    """Extract one context condition with hash caching and domain checkpoints."""

    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode: {context_mode}")
    root = repository_root()
    output_directory = root / "artifacts/activations" / bundle.alias
    destination = output_directory / f"{context_mode}.npz"
    expected_order = prefix_frame["example_id"].tolist()
    dataset_hashes = prefix_frame["dataset_hash"].drop_duplicates().tolist()
    if len(dataset_hashes) != 1:
        raise RuntimeError("Prefix table must contain exactly one dataset hash.")
    input_metadata = _input_metadata(
        bundle, context_mode, selected_layers, config["model_runtime"]
    )
    checkpoint_metadata = {
        **input_metadata,
        "dataset_hash": dataset_hashes[0],
        "hidden_size": int(model_metadata["hidden_size"]),
    }
    records = load_activation_checkpoint(destination, checkpoint_metadata)
    unexpected = sorted(set(records) - set(expected_order))
    if unexpected:
        raise RuntimeError(
            f"Checkpoint contains examples absent from prefixes: {unexpected[:5]}"
        )

    activation_cache: dict[str, tuple[dict[int, np.ndarray], dict[str, Any]]] = {}
    for record in records.values():
        activation_cache.setdefault(
            record["input_hash"],
            (
                record["vectors"],
                {
                    "original_token_count": record["input_token_count"],
                    "final_token_id": record["readout_token_id"],
                },
            ),
        )

    for domain in config["domains"]:
        domain_frame = prefix_frame[prefix_frame["domain"] == domain]
        if domain_frame.empty:
            raise RuntimeError(f"Prefix table contains no rows for domain {domain}.")
        new_forwards = 0
        for _, row in domain_frame.iterrows():
            example_id = row["example_id"]
            if example_id in records:
                continue
            messages = _messages_for_row(row, context_mode)
            rendered = render_chat(
                bundle.tokenizer,
                messages,
                enable_thinking=config["model_runtime"]["enable_thinking"],
                add_generation_prompt=config["model_runtime"]["add_generation_prompt"],
            )
            input_hash = hash_rendered_input(rendered, input_metadata)
            cached = activation_cache.get(input_hash)
            if cached is None:
                vectors, extraction_metadata = extract_selected_activations(
                    bundle.model,
                    bundle.tokenizer,
                    rendered,
                    selected_layers,
                    int(config["model_runtime"]["max_length"]),
                )
                activation_cache[input_hash] = (vectors, extraction_metadata)
                new_forwards += 1
            else:
                vectors, extraction_metadata = cached
            records[example_id] = {
                "example_id": example_id,
                "input_hash": input_hash,
                "input_token_count": int(extraction_metadata["original_token_count"]),
                "readout_token_id": int(extraction_metadata["final_token_id"]),
                "vectors": {
                    layer: np.asarray(vector, dtype=np.float32)
                    for layer, vector in vectors.items()
                },
            }

        completed_domains = [
            candidate
            for candidate in config["domains"]
            if set(
                prefix_frame[prefix_frame["domain"] == candidate]["example_id"]
            ).issubset(records)
        ]
        save_activation_checkpoint(
            records,
            expected_order,
            selected_layers,
            destination,
            {**checkpoint_metadata, "completed_domains": completed_domains},
        )
        sync_directory_to_persistent(output_directory)
        print(
            f"{bundle.alias} {context_mode}: {domain} complete; "
            f"{new_forwards} new forwards, {len(records)}/{len(expected_order)} rows."
        )

    validate_activation_artifact(
        destination,
        expected_order,
        selected_layers,
        int(model_metadata["hidden_size"]),
    )
    _assert_pair_invariants(prefix_frame, records, context_mode)
    return destination


def run_bulk_activation_extraction(
    config: dict[str, Any],
    model_alias: str,
) -> list[Path]:
    """Load one frozen checkpoint and extract both registered context modes."""

    root = repository_root()
    ensure_output_directories(root)
    prefix_path = Path(config["project"]["processed_path"])
    if not prefix_path.is_absolute():
        prefix_path = root / prefix_path
    if not prefix_path.is_file():
        raise FileNotFoundError(f"Prefix table not found: {prefix_path}")
    prefix_frame = pd.read_parquet(prefix_path)
    scenario_count = int(prefix_frame["scenario_id"].nunique())
    prefix_errors = validate_prefix_dataframe(prefix_frame, scenario_count)
    if prefix_errors:
        raise RuntimeError("Invalid prefix table:\n- " + "\n- ".join(prefix_errors))

    frozen_hash_path = root / "data/FROZEN_DATASET.sha256"
    frozen_hash = frozen_hash_path.read_text(encoding="utf-8").strip()
    table_hashes = prefix_frame["dataset_hash"].drop_duplicates().tolist()
    if table_hashes != [frozen_hash]:
        raise RuntimeError(
            "Prefix dataset hash does not match data/FROZEN_DATASET.sha256."
        )

    model_spec = get_model_spec(config, model_alias)
    runtime = config["model_runtime"]
    verify_smoke_test_artifact(
        root,
        model_alias,
        model_spec["model_id"],
        model_spec["revision"],
    )
    revision = resolve_model_revision(model_spec["model_id"], model_spec["revision"])
    tokenizer = load_tokenizer(model_spec["model_id"], revision)
    model = load_model(
        model_spec["model_id"],
        revision,
        runtime["dtype"],
        runtime["device_map"],
    )
    bundle = ModelBundle(
        alias=model_alias,
        model_id=model_spec["model_id"],
        revision=revision,
        tokenizer=tokenizer,
        model=model,
    )
    try:
        model_metadata = collect_model_metadata(
            model,
            tokenizer,
            bundle.model_id,
            bundle.revision,
            bundle.alias,
            runtime["dtype"],
            runtime["device_map"],
        )
        selected_layers = selected_layer_indices(
            model_metadata["num_hidden_layers"],
            runtime["selected_depth_fractions"],
        )
        model_metadata.update(
            {
                "selected_layers": selected_layers,
                "enable_thinking": runtime["enable_thinking"],
                "dataset_hash": frozen_hash,
            }
        )
        metadata_path = root / "artifacts/model_metadata" / f"{model_alias}.json"
        metadata_path.write_text(
            json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        outputs = [
            run_activation_condition(
                bundle,
                prefix_frame,
                context_mode,
                config,
                model_metadata,
                selected_layers,
            )
            for context_mode in CONTEXT_MODES
        ]
        sync_directory_to_persistent(metadata_path.parent)
        print(f"Activation extraction passed for {model_alias}.")
        return outputs
    finally:
        release_model(bundle)

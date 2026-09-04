"""Deterministic Turn-4 response generation with resumable checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .environment import sync_directory_to_persistent
from .model_loading import ModelBundle, model_input_device
from .rendering import hash_rendered_input, render_chat


def generate_one(
    model, tokenizer, rendered: str, config: dict[str, Any]
) -> dict[str, Any]:
    import torch

    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model_input_device(model)) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=bool(config["do_sample"]),
            use_cache=bool(config["use_cache"]),
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, input_length:]
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "generation_seconds": time.perf_counter() - started,
        "generated_token_count": int(new_tokens.shape[-1]),
        "finish_reason": (
            "eos"
            if len(new_tokens) and int(new_tokens[-1]) == tokenizer.eos_token_id
            else "length_or_stop"
        ),
    }


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["response_id"]] = record
    return records


def _write_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(records[key], sort_keys=True, ensure_ascii=False) + "\n"
            for key in sorted(records)
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def generate_turn4_responses(
    bundle: ModelBundle, prefix_frame: pd.DataFrame, config: dict[str, Any]
) -> Path:
    """Generate one full-history response for every primary Turn-4 row."""

    rows = prefix_frame[prefix_frame["is_primary_t4"]].copy()
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/responses" / f"{bundle.alias}.jsonl"
    records = _read_jsonl(output)
    dataset_hash = rows["dataset_hash"].iloc[0]
    if any(
        record.get("model_revision") != bundle.revision
        or record.get("dataset_hash") != dataset_hash
        for record in records.values()
    ):
        raise RuntimeError("Existing response checkpoint has incompatible metadata.")
    generation = config["generation"]
    runtime = config["model_runtime"]
    hash_metadata = {
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "context_mode": "full_history",
        "generation": generation,
    }
    for domain in config["domains"]:
        for _, row in rows[rows["domain"] == domain].iterrows():
            response_id = f"{bundle.alias}__{row['example_id']}"
            if response_id in records:
                continue
            messages = json.loads(row["messages_full_json"])
            rendered = render_chat(
                bundle.tokenizer,
                messages,
                runtime["enable_thinking"],
                runtime["add_generation_prompt"],
            )
            result = generate_one(bundle.model, bundle.tokenizer, rendered, generation)
            records[response_id] = {
                "response_id": response_id,
                "example_id": row["example_id"],
                "scenario_id": row["scenario_id"],
                "domain": row["domain"],
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "input_hash": hash_rendered_input(rendered, hash_metadata),
                "generation_config": generation,
                "raw_response": result["text"],
                "finish_reason": result["finish_reason"],
                "generation_seconds": result["generation_seconds"],
                "generated_token_count": result["generated_token_count"],
                "publication_redaction_required": False,
                "dataset_hash": row["dataset_hash"],
            }
        _write_jsonl(output, records)
        sync_directory_to_persistent(output.parent)
        print(f"{bundle.alias} responses: {domain} complete; {len(records)}/64 rows.")
    if len(records) != 64:
        raise RuntimeError(f"Expected 64 responses; found {len(records)}.")
    return output

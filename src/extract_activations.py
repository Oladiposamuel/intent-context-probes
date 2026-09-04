"""Assistant-decision-position activation extraction."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .model_loading import model_input_device
from .rendering import inspect_readout_token


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
            f"Input has {original_length} tokens and would exceed max_length={max_length}."
        )

    inputs = {key: value.to(model_input_device(model)) for key, value in untruncated.items()}
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

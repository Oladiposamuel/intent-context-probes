"""Frozen-model loading, revision resolution and metadata capture."""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    alias: str
    model_id: str
    revision: str
    tokenizer: Any
    model: Any


def resolve_model_revision(model_id: str) -> str:
    """Resolve the immutable Hugging Face commit SHA used for a run."""

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(model_id)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {model_id}.")
    return info.sha


def load_tokenizer(model_id: str, revision: str):
    """Load the official tokenizer without executing repository code."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=False,
    )


def load_model(
    model_id: str,
    revision: str,
    dtype_name: str,
    device_map: str,
):
    """Load a frozen causal language model in the configured precision."""

    import torch
    from transformers import AutoModelForCausalLM

    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=dtype_by_name[dtype_name],
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_input_device(model):
    """Return the device on which token IDs should be placed."""

    return model.get_input_embeddings().weight.device


def collect_model_metadata(
    model,
    tokenizer,
    model_id: str,
    revision: str,
    alias: str,
    dtype_name: str,
    device_map: str,
) -> dict[str, Any]:
    """Record the runtime facts needed to reproduce a model forward pass."""

    chat_template = tokenizer.chat_template or ""
    return {
        "alias": alias,
        "model_id": model_id,
        "revision": revision,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "vocabulary_size": int(len(tokenizer)),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "dtype_requested": dtype_name,
        "device_map_requested": device_map,
        "training": bool(model.training),
        "parameters_require_grad": any(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }


def release_model(bundle_or_model) -> None:
    """Release one checkpoint before loading the next checkpoint."""

    model = getattr(bundle_or_model, "model", bundle_or_model)
    if hasattr(bundle_or_model, "model"):
        bundle_or_model.model = None
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

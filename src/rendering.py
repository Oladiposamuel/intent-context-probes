"""Shared chat rendering and readout-token inspection."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def validate_messages(messages: list[dict[str, str]]) -> None:
    """Reject malformed or non-alternating controlled histories."""

    if not messages:
        raise ValueError("At least one chat message is required.")
    if messages[-1].get("role") != "user":
        raise ValueError("The activation prefix must end with a user message.")
    expected = "user"
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(f"Message {index} must contain role and content only.")
        if message["role"] != expected:
            raise ValueError(
                f"Message {index} has role {message['role']!r}; expected {expected!r}."
            )
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"Message {index} has empty content.")
        expected = "assistant" if expected == "user" else "user"


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact prefix shared by extraction and generation."""

    validate_messages(messages)
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise RuntimeError(
            "The tokenizer chat template did not accept enable_thinking. "
            "Do not silently change templates; verify the Transformers version."
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Chat template returned an empty or non-text result.")
    return rendered


def hash_rendered_input(rendered_text: str, metadata: dict[str, Any]) -> str:
    """Hash the text and every setting that defines an activation input."""

    payload = {
        "rendered_text": rendered_text,
        "metadata": metadata,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_readout_token(tokenizer, input_ids, count: int = 10) -> dict[str, Any]:
    """Decode the final input tokens used at the assistant-decision readout."""

    ids = input_ids[0].detach().cpu().tolist()
    tail = ids[-count:]
    return {
        "token_count": len(ids),
        "final_token_id": ids[-1],
        "tail_token_ids": tail,
        "tail_decoded": tokenizer.decode(tail, skip_special_tokens=False),
    }


def plain_text_transcript(messages: list[dict[str, str]]) -> str:
    """Return a readable transcript for inspection and future text baselines."""

    validate_messages(messages)
    return "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )

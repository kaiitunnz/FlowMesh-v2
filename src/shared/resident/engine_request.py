"""Build the OpenAI-compatible engine request from a boundary payload."""

import json
from typing import Any


def chat_body(request_payload: str | None, model: str) -> dict[str, Any]:
    """Build the OpenAI chat request from a boundary payload.

    A payload that is already a chat request (a JSON object carrying ``messages``) is
    forwarded faithfully with the replica's model pinned, preserving its system and
    multi-turn messages and sampling parameters; a bare prompt is wrapped as one user
    message. A payload naming a single ``prompt``/``input``/``content`` field is treated
    as that prompt.
    """
    parsed: Any = None
    if request_payload:
        try:
            parsed = json.loads(request_payload)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
        return {**parsed, "model": model}
    if isinstance(parsed, dict):
        for key in ("prompt", "input", "content"):
            if isinstance(value := parsed.get(key), str):
                return {
                    "model": model,
                    "messages": [{"role": "user", "content": value}],
                }
    prompt = request_payload or ""
    return {"model": model, "messages": [{"role": "user", "content": prompt}]}


def embeddings_body(request_payload: str | None, model: str) -> dict[str, Any]:
    """Build the OpenAI embeddings request from a boundary payload.

    A payload that is already an embeddings request (a JSON object carrying ``input``)
    is forwarded faithfully with the replica's model pinned; a bare string or a payload
    naming a single ``input``/``text``/``content`` field becomes the one input to embed.
    """
    parsed: Any = None
    if request_payload:
        try:
            parsed = json.loads(request_payload)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    if isinstance(parsed, dict) and "input" in parsed:
        return {**parsed, "model": model}
    if isinstance(parsed, dict):
        for key in ("text", "content"):
            if isinstance(value := parsed.get(key), str):
                return {"model": model, "input": [value]}
    return {"model": model, "input": [request_payload or ""]}

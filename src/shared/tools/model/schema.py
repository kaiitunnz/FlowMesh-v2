"""The external managed-model request schema and its integrity digest.

The control path derives a canonical ``ModelRequest`` from an agent's managed-model
boundary and commits to it with ``model_request_digest``; the worker executor recomputes
the digest over the delivered request before any provider egress, so the raw request
is worker-private while control holds only the digest.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

# The reserved interface of a deferred managed-model invocation, distinct from a
# fabric-served tool interface. Exact routing keys on this exact value.
MODEL_INTERFACE = "model"


class ModelRequest(BaseModel):
    """The canonical external-model request the worker egresses.

    ``url`` comes from the activation's pinned binding (credential-free). ``body`` is
    the exact chat-completions payload posted to it — the model, the messages, and any
    tools the facade injected. The credential is resolved at the worker from the permit
    or the local environment, never here.
    """

    model_config = ConfigDict(frozen=True)

    interface: str
    url: str
    body: dict[str, Any]


class ModelToolCall(BaseModel):
    """One tool call in a model's chat completion, carried through the facade."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: str


class ModelCompletion(BaseModel):
    """A model's full assistant message: its text and any tool calls it emitted.

    The held-turn facade needs the whole message, not just the text, so it can surface
    the model's tool calls and capture the fabric-facade ones as boundaries.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()


def parse_model_request(payload: str | None, *, url: str, model: str) -> ModelRequest:
    """Build a ``ModelRequest`` from a single-prompt boundary payload and binding.

    The prompt is the ``prompt``/``input``/``content`` field of a JSON payload, or the
    bare payload string, wrapped as one user message. This is the deferred-boundary
    form; a held facade builds its own multi-message body. Control never sees either.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _extract_prompt(payload)}],
    }
    return ModelRequest(interface=MODEL_INTERFACE, url=url, body=body)


def _extract_prompt(payload: str | None) -> str:
    if not payload:
        return ""
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if isinstance(parsed, dict):
        for key in ("prompt", "input", "content"):
            if isinstance(value := parsed.get(key), str):
                return value
    return payload


def model_request_digest(interface: str, url: str, body: dict[str, Any]) -> str:
    """A canonical integrity digest over the request the fence commits to.

    The worker recomputes it over the delivered request and rejects a mismatch, so an
    altered request or an altered digest fails the fence before any provider call. The
    body is serialized canonically so the propose-side and fence-side digests agree.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    raw = f"{interface}\x00{url}\x00{canonical}".encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MODEL_INTERFACE",
    "ModelCompletion",
    "ModelRequest",
    "ModelToolCall",
    "model_request_digest",
    "parse_model_request",
]

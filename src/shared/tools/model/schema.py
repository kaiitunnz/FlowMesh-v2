"""The external managed-model request schema and its integrity digest.

The control path derives a canonical ``ModelRequest`` from an agent's managed-model
boundary and commits to it with ``model_request_digest``; the worker executor recomputes
the digest over the delivered request before any provider egress, so the raw request
stays worker-private while control holds only the digest.
"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict

# The reserved interface of a deferred managed-model invocation, distinct from a
# fabric-served tool interface. Exact routing keys on this exact value.
MODEL_INTERFACE = "model"


class ModelRequest(BaseModel):
    """The canonical external-model request the worker egresses.

    ``url`` and ``model`` come from the activation's pinned binding (credential-free);
    the credential is resolved at the worker from its local environment, never here.
    """

    model_config = ConfigDict(frozen=True)

    interface: str
    url: str
    model: str
    prompt: str


def parse_model_request(payload: str | None, *, url: str, model: str) -> ModelRequest:
    """Build a canonical ``ModelRequest`` from a boundary payload and pinned binding.

    The prompt is the ``prompt``/``input``/``content`` field of a JSON payload, or the
    bare payload string. This is the canonical form the origin worker digests and
    egresses against; the control plane never sees the payload.
    """
    return ModelRequest(
        interface=MODEL_INTERFACE, url=url, model=model, prompt=_extract_prompt(payload)
    )


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


def model_request_digest(interface: str, url: str, model: str, prompt: str) -> str:
    """A canonical integrity digest over the request the fence commits to.

    The worker executor recomputes it over the delivered request and rejects a mismatch,
    so an altered request or an altered digest fails the fence before any provider call.
    """
    raw = f"{interface}\x00{url}\x00{model}\x00{prompt}".encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MODEL_INTERFACE",
    "ModelRequest",
    "model_request_digest",
    "parse_model_request",
]

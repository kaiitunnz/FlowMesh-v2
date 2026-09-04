"""The web-search tool's request schema and its integrity digest.

The control path derives a bounds-shaped ``ToolRequest`` from an agent's ``search/v1``
boundary and commits to it with ``tool_request_digest``; the worker executor recomputes
the digest over the delivered request before any provider egress.
"""

import hashlib

from pydantic import BaseModel, ConfigDict

# The reserved fabric-served tool interface. Exact routing keys on this exact value.
SEARCH_INTERFACE = "search/v1"

# The keyless web-search backend selected when no provider is configured.
DEFAULT_SEARCH_PROVIDER = "duckduckgo"


class ToolRequest(BaseModel):
    """The parsed, bounds-shaped request the control path derived from a boundary."""

    model_config = ConfigDict(frozen=True)

    interface: str
    query: str
    max_results: int


def tool_request_digest(interface: str, query: str, max_results: int) -> str:
    """A canonical integrity digest over the bounded request the fence commits to.

    The worker executor recomputes it over the delivered request and rejects a mismatch,
    so an altered request or an altered digest fails the fence before any provider call.
    """
    raw = f"{interface}\x00{query}\x00{max_results}".encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DEFAULT_SEARCH_PROVIDER",
    "SEARCH_INTERFACE",
    "ToolRequest",
    "tool_request_digest",
]

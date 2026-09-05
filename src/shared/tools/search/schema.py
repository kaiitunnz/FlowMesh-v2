"""The web-search tool's request schema and its integrity digest.

The control path derives a bounds-shaped ``ToolRequest`` from an agent's ``search/v1``
boundary and commits to it with ``tool_request_digest``; the worker executor recomputes
the digest over the delivered request before any provider egress.
"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict

# The reserved fabric-served tool interface. Exact routing keys on this exact value.
SEARCH_INTERFACE = "search/v1"

# The keyless web-search backend selected when no provider is configured.
DEFAULT_SEARCH_PROVIDER = "duckduckgo"

# The result count a request defaults to when the model names none.
DEFAULT_SEARCH_MAX_RESULTS = 5


class ToolRequest(BaseModel):
    """The parsed, bounds-shaped request the control path derived from a boundary."""

    model_config = ConfigDict(frozen=True)

    interface: str
    query: str
    max_results: int


def parse_search_request(
    payload: str | None, *, default_max_results: int = DEFAULT_SEARCH_MAX_RESULTS
) -> ToolRequest:
    """Parse a raw ``search/v1`` request payload into a bounds-shaped ``ToolRequest``.

    Accepts a JSON object (``{"query", "max_results"}``) or a bare query string. An
    empty or malformed payload yields an empty query, which the caller handles. This
    is the canonical parse the origin worker digests and executes against; the control
    plane never sees the raw payload.
    """
    if not payload:
        return ToolRequest(
            interface=SEARCH_INTERFACE, query="", max_results=default_max_results
        )
    try:
        args = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return ToolRequest(
            interface=SEARCH_INTERFACE,
            query=payload.strip(),
            max_results=default_max_results,
        )
    if not isinstance(args, dict):
        return ToolRequest(
            interface=SEARCH_INTERFACE, query="", max_results=default_max_results
        )
    query = args.get("query")
    requested = args.get("max_results")
    n = int(requested) if isinstance(requested, int) else default_max_results
    return ToolRequest(
        interface=SEARCH_INTERFACE,
        query=query.strip() if isinstance(query, str) else "",
        max_results=max(1, n),
    )


def tool_request_digest(interface: str, query: str, max_results: int) -> str:
    """A canonical integrity digest over the bounded request the fence commits to.

    The worker executor recomputes it over the delivered request and rejects a mismatch,
    so an altered request or an altered digest fails the fence before any provider call.
    """
    raw = f"{interface}\x00{query}\x00{max_results}".encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DEFAULT_SEARCH_MAX_RESULTS",
    "DEFAULT_SEARCH_PROVIDER",
    "SEARCH_INTERFACE",
    "ToolRequest",
    "parse_search_request",
    "tool_request_digest",
]

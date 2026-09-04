"""The provider-egress surface for a fabric-served external tool.

It egresses only within a server-issued ``ToolOperationEnvelope``, refusing an interface
it does not serve or a request beyond the issued budget, and maps a provider fault to a
typed ``ToolOutcome``. It runs in whichever process actually egresses — a worker
executor, or the transitional in-server relay path.
"""

import logging

from .providers import (
    SearchProvider,
    SearchQuotaExceeded,
    SearchResult,
    SearchTimeout,
    SearchUnavailable,
)
from .schema import (
    SEARCH_INTERFACE,
    ToolOperationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
    ToolRequest,
)

_SERVED_INTERFACES = frozenset({SEARCH_INTERFACE})


class ExternalToolSidecar:
    """The surface that performs external-tool egress under an envelope."""

    def __init__(
        self, provider: SearchProvider, logger: logging.Logger | None = None
    ) -> None:
        self._provider = provider
        self._log = logger or logging.getLogger("external-tool-sidecar")

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> ToolOutcome:
        if envelope.interface not in _SERVED_INTERFACES:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value=f"the sidecar serves no interface {envelope.interface!r}",
            )
        if request.interface != envelope.interface:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the request interface is outside the issued envelope",
            )
        if request.max_results > envelope.max_results:
            return ToolOutcome(
                status=ToolOutcomeStatus.QUOTA,
                value="the request exceeds the issued operation budget",
            )
        try:
            results = self._provider.search(
                request.query,
                max_results=request.max_results,
                timeout_sec=envelope.timeout_sec,
            )
        except SearchTimeout:
            return ToolOutcome(
                status=ToolOutcomeStatus.TIMEOUT, value="the web search timed out"
            )
        except SearchQuotaExceeded:
            return ToolOutcome(
                status=ToolOutcomeStatus.QUOTA, value="the search provider rate-limited"
            )
        except SearchUnavailable:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the search provider was unreachable",
            )
        return self._normalize(request.query, results, envelope.result_char_cap)

    @staticmethod
    def _normalize(
        query: str, results: list[SearchResult], char_cap: int
    ) -> ToolOutcome:
        if not results:
            return ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS, value=f"No results for {query!r}."
            )
        blocks: list[str] = []
        provenance: list[dict[str, str]] = []
        for i, r in enumerate(results, 1):
            blocks.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
            provenance.append({"title": r.title, "url": r.url})
        return ToolOutcome(
            status=ToolOutcomeStatus.SUCCESS,
            value="\n\n".join(blocks)[:char_cap],
            provenance=tuple(provenance),
        )


__all__ = ["ExternalToolSidecar"]

"""Execution-locality machinery for fabric-served external tools.

The authoritative control path (``FabricToolBroker``) checks authority, draws down the
episode quota, and issues a bounded ``ToolOperationEnvelope`` naming the single
interface and result budget a request is authorized for. It then hands the operation to
a selected ``ExecutionLocalityAdapter``: a ``server_relay`` that egresses in-server
against the deployment provider, or a ``worker_sidecar`` that carries the operation to a
fabric ``ExternalToolSidecar``. The sidecar is the fabric-controlled surface that
performs the provider egress and enforces the envelope, refusing any interface it does
not serve or a request beyond its issued budget; the agent harness never egresses. A
deployment policy selects the locality per provider, with server relay the default.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from ..orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    ToolOutcome,
    ToolOutcomeStatus,
)
from .search_providers import (
    SearchProvider,
    SearchQuotaExceeded,
    SearchResult,
    SearchTimeout,
    SearchUnavailable,
)

if TYPE_CHECKING:
    from ..config import WebSearchConfig

_SERVED_INTERFACES = frozenset({SEARCH_INTERFACE})


class EgressLocality(StrEnum):
    """Where an approved external-tool operation executes."""

    SERVER_RELAY = "server_relay"
    WORKER_SIDECAR = "worker_sidecar"


class ToolRequest(BaseModel):
    """The parsed, bounds-shaped request the control path derived from a boundary."""

    model_config = ConfigDict(frozen=True)

    interface: str
    query: str
    max_results: int


class ToolOperationEnvelope(BaseModel):
    """A server-issued authorization for exactly one bounded external-tool operation.

    The control path issues it after the authority and quota checks; the execution
    surface egresses only within it. ``idempotency_key`` is the fabric dedupe authority
    a re-drive reuses; the result bounds cap the single authorized operation.
    """

    model_config = ConfigDict(frozen=True)

    interface: str
    idempotency_key: str | None
    max_results: int
    timeout_sec: float
    result_char_cap: int


class ExternalToolSidecar:
    """The fabric-controlled surface that performs external-tool egress under an
    envelope.

    It egresses only within the server-issued ``ToolOperationEnvelope``, refusing an
    interface it does not serve or a request beyond the issued result budget, and maps a
    provider fault to a typed outcome.
    """

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


# Carriage from the control path to an execution surface for one bounded operation.
ExecutionTransport = Callable[[ToolOperationEnvelope, ToolRequest], ToolOutcome]


class ExecutionLocalityAdapter(Protocol):
    """A selected execution locality for one approved external-tool operation."""

    locality: EgressLocality

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> ToolOutcome: ...


class ServerRelayAdapter:
    """Egresses an approved operation in-server against the deployment provider.

    The credential-bearing provider stays on the server-to-upstream path; no credential
    material crosses to a worker.
    """

    locality = EgressLocality.SERVER_RELAY

    def __init__(self, sidecar: ExternalToolSidecar) -> None:
        self._sidecar = sidecar

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> ToolOutcome:
        return self._sidecar.execute(envelope, request)


class WorkerSidecarAdapter:
    """Hands an approved operation to a worker-sidecar surface over its carriage."""

    locality = EgressLocality.WORKER_SIDECAR

    def __init__(self, transport: ExecutionTransport) -> None:
        self._transport = transport

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> ToolOutcome:
        return self._transport(envelope, request)


class ColocatedSidecarCarriage:
    """Carriage to an ``ExternalToolSidecar`` co-located with the control plane."""

    def __init__(self, sidecar: ExternalToolSidecar) -> None:
        self._sidecar = sidecar

    def __call__(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> ToolOutcome:
        return self._sidecar.execute(envelope, request)


class EgressLocalityPolicy:
    """Selects the execution locality for an external-tool operation under deployment
    policy.

    Server relay is the default; a worker sidecar is selected when the deployment
    configures it.
    """

    def __init__(
        self,
        config: "WebSearchConfig",
        server: ExecutionLocalityAdapter,
        worker: ExecutionLocalityAdapter,
    ) -> None:
        self._cfg = config
        self._server = server
        self._worker = worker

    def select(self) -> ExecutionLocalityAdapter:
        if self._cfg.egress_locality == EgressLocality.WORKER_SIDECAR:
            return self._worker
        return self._server

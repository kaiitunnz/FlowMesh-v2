"""Execution-locality machinery for fabric-served external tools.

The authoritative control path (``FabricToolBroker``) checks authority, draws down the
episode quota, and issues a bounded ``ToolOperationEnvelope`` naming the one interface
and result budget a request is authorized for. It then hands the operation to a selected
``ExecutionLocalityAdapter``: a ``server_relay`` that egresses in-server against the
deployment provider (transitional), or a ``worker_sidecar`` carrying the operation to a
worker executor. The shared schemas and the provider egress surface live in
``shared.tools``; a deployment policy selects the locality per provider, relay the
default.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from shared.tools.egress import ExternalToolSidecar
from shared.tools.schema import (
    RemoteToolOperationEnvelope as RemoteToolOperationEnvelope,
)
from shared.tools.schema import (
    ToolOperationEnvelope,
    ToolOutcome,
    ToolRequest,
)
from shared.tools.schema import tool_request_digest as tool_request_digest

if TYPE_CHECKING:
    from ..config import WebSearchConfig


class EgressLocality(StrEnum):
    """Where an approved external-tool operation executes."""

    SERVER_RELAY = "server_relay"
    WORKER_SIDECAR = "worker_sidecar"


class AmbiguousDelivery:
    """A nonterminal carriage result: the operation may have egressed but its reply was
    lost.

    It is distinct from a ``ToolOutcome``: a lost acknowledgement, reverse session,
    response, or post-write stream after possible worker/provider execution is never
    coerced into a terminal ``UNAVAILABLE``/``TIMEOUT`` value because carriage lost it.
    The control path holds the durable boundary pending and re-drives the logical
    operation under its ``idempotency_key``, within its bounded retry budget.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


# Carriage from the control path to an execution surface for one bounded operation. A
# terminal ``ToolOutcome`` settles the boundary; an ``AmbiguousDelivery`` holds it
# pending for a same-``idempotency_key`` re-drive.
CarriageResult = ToolOutcome | AmbiguousDelivery
ExecutionTransport = Callable[[ToolOperationEnvelope, ToolRequest], CarriageResult]


class ExecutionLocalityAdapter(Protocol):
    """A selected execution locality for one approved external-tool operation."""

    locality: EgressLocality

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> CarriageResult: ...


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
    ) -> CarriageResult:
        return self._sidecar.execute(envelope, request)


class WorkerSidecarAdapter:
    """Hands an approved operation to a worker-sidecar surface over its carriage."""

    locality = EgressLocality.WORKER_SIDECAR

    def __init__(self, transport: ExecutionTransport) -> None:
        self._transport = transport

    def execute(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> CarriageResult:
        return self._transport(envelope, request)


class ColocatedSidecarCarriage:
    """Carriage to an ``ExternalToolSidecar`` co-located with the control plane."""

    def __init__(self, sidecar: ExternalToolSidecar) -> None:
        self._sidecar = sidecar

    def __call__(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> CarriageResult:
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

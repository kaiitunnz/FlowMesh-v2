"""Server-side control of the native resident data path.

The Admission controller stays authoritative over the claim FSM and the fence, but the
resident bytes never cross the server. This transport pokes the origin node's deputy
over the node-command seam — bind a sidecar, deliver a bootstrap, stream under the
fence, cancel — and returns the deputy's control result. The stream, backpressure, and
cancellation ride the data-direct deputy-to-sidecar channel, not this seam.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from shared.schemas.command import CommandType

from ..network.state import ResolvedRoute, RouteObservationOutcome, Transport
from .state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization

# Sends one node command and returns its result data, raising on a failed command.
ExecNodeCmd = Callable[[str, CommandType, dict[str, Any]], Awaitable[dict[str, Any]]]


class NativeTransportError(RuntimeError):
    """A node command that drives the resident data path failed or was unreachable."""


@dataclass(frozen=True)
class BootstrapReply:
    """The bootstrap phase result carried back from the origin deputy."""

    acked: bool
    rejection: str | None
    uncertain: bool
    selected_transport: Transport | None
    observations: list[tuple[Transport, RouteObservationOutcome]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class StreamReply:
    """The stream phase result; a non-ok post-acceptance result is uncertain."""

    ok: bool
    completion: str | None
    rejection: str | None


class NativeTransport:
    """Drives the origin node's deputy and replica sidecars over node commands."""

    def __init__(self, exec_cmd: ExecNodeCmd) -> None:
        self._exec = exec_cmd

    async def bind_sidecar(
        self,
        node_id: str,
        *,
        replica_id: str,
        incarnation: int,
        listener_generation: int,
        route: str,
        engine: ReplicaEndpoint,
    ) -> tuple[str, int]:
        """Bind a replica's claim-gated sidecar on its node; return host and port."""
        data = await self._exec(
            node_id,
            CommandType.BIND_RESIDENT_SIDECAR,
            {
                "replica_id": replica_id,
                "incarnation": incarnation,
                "listener_generation": listener_generation,
                "route": route,
                "engine": {
                    "base_url": engine.base_url,
                    "model": engine.model,
                    "api_key": engine.api_key,
                },
            },
        )
        return str(data["host"]), int(data["port"])

    async def unbind_sidecar(self, node_id: str, replica_id: str) -> None:
        await self._exec(
            node_id,
            CommandType.UNBIND_RESIDENT_SIDECAR,
            {"replica_id": replica_id},
        )

    async def bootstrap(
        self,
        node_id: str,
        *,
        session_id: str,
        route: ResolvedRoute,
        handoff: AdmissionHandoff,
        request_payload: str | None,
    ) -> BootstrapReply:
        data = await self._exec(
            node_id,
            CommandType.DELIVER_RESIDENT_BOOTSTRAP,
            {
                "session_id": session_id,
                "resolved_route": route.model_dump(mode="json"),
                "handoff": handoff.model_dump(mode="json"),
                "request": request_payload,
            },
        )
        selected = data.get("selected_transport")
        return BootstrapReply(
            acked=bool(data.get("acked")),
            rejection=data.get("rejection"),
            uncertain=bool(data.get("uncertain")),
            selected_transport=Transport(selected) if selected else None,
            observations=[
                (Transport(o["transport"]), RouteObservationOutcome(o["outcome"]))
                for o in data.get("observations", [])
            ],
        )

    async def stream(
        self, node_id: str, *, session_id: str, auth: RouteAuthorization
    ) -> StreamReply:
        data = await self._exec(
            node_id,
            CommandType.DELIVER_RESIDENT_STREAM,
            {"session_id": session_id, "auth": auth.model_dump(mode="json")},
        )
        return StreamReply(
            ok=bool(data.get("ok")),
            completion=data.get("completion"),
            rejection=data.get("rejection"),
        )

    async def cancel(
        self, node_id: str, *, session_id: str, auth: RouteAuthorization
    ) -> None:
        await self._exec(
            node_id,
            CommandType.DELIVER_RESIDENT_CANCEL,
            {"session_id": session_id, "auth": auth.model_dump(mode="json")},
        )

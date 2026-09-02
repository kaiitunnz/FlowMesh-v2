"""Supervisor-side resident invocation deputy and sidecar registry.

On the origin node this runs one invocation deputy, holding a session across the
bootstrap and stream control pokes so the response streams over one data-direct
connection. On the replica node it binds and drops the per-replica claim-gated sidecar.
Both roles live in one node's supervisor process; a single-node deployment hosts both.

The control plane pokes it over the node-command seam; the resident bytes never traverse
that seam — they ride the data-direct channel between this deputy and the sidecar.
"""

import logging
from typing import Any

from ...network.state import ResolvedRoute
from ...resident.deputy import ResidentInvocationDeputy
from ...resident.sidecar import SidecarClaimGate
from ...resident.sidecar_server import (
    EngineOpen,
    HttpEngineDelivery,
    ResidentSidecarListener,
    ResidentSidecarServer,
)
from ...resident.state import AdmissionHandoff, ReplicaEndpoint, RouteAuthorization


class ResidentDeputyService:
    """One node's resident invocation deputy plus its bound sidecars."""

    def __init__(
        self,
        *,
        connect_budget_sec: float,
        engine_timeout_sec: float = 300.0,
        engine_open: EngineOpen | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._deputy = ResidentInvocationDeputy(
            connect_budget_sec=connect_budget_sec, logger=logger
        )
        self._sidecars: dict[str, ResidentSidecarListener] = {}
        self._engine_open = engine_open or HttpEngineDelivery(
            timeout_sec=engine_timeout_sec
        )
        self._logger = logger

    async def bind_sidecar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind (or rebind) the claim-gated sidecar for a replica incarnation."""
        replica_id = str(payload["replica_id"])
        engine = payload["engine"]
        endpoint = ReplicaEndpoint(
            base_url=str(engine["base_url"]),
            model=str(engine.get("model") or ""),
            api_key=engine.get("api_key"),
        )
        gate = SidecarClaimGate(
            replica_id=replica_id,
            incarnation=int(payload["incarnation"]),
            listener_generation=int(payload["listener_generation"]),
        )
        server = ResidentSidecarServer(
            gate=gate,
            endpoint=endpoint,
            engine_open=self._engine_open,
            logger=self._logger,
        )
        listener = ResidentSidecarListener(server, route=str(payload["route"]))
        await self.unbind_sidecar(replica_id)
        host, port = await listener.start()
        self._sidecars[replica_id] = listener
        return {"bound": True, "host": host, "port": port}

    async def unbind_sidecar(self, replica_id: str) -> dict[str, Any]:
        """Drop a replica's sidecar; absent is a no-op."""
        listener = self._sidecars.pop(replica_id, None)
        if listener is not None:
            await listener.stop()
        return {"unbound": True}

    async def bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Phase 1: deliver the bootstrap and report the enqueue acknowledgement."""
        route = ResolvedRoute.model_validate(payload["resolved_route"])
        handoff = AdmissionHandoff.model_validate(payload["handoff"])
        result = await self._deputy.bootstrap(
            str(payload["session_id"]), route, handoff, payload.get("request")
        )
        return {
            "acked": result.acked,
            "rejection": result.rejection,
            "uncertain": result.uncertain,
            "selected_transport": (
                result.selected_transport.value
                if result.selected_transport is not None
                else None
            ),
            "observations": [
                {"transport": transport.value, "outcome": outcome.value}
                for transport, outcome in result.observations
            ],
        }

    async def stream(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Phase 2: present the authorization and return the assembled completion."""
        auth = RouteAuthorization.model_validate(payload["auth"])
        result = await self._deputy.stream(str(payload["session_id"]), auth)
        return {
            "ok": result.ok,
            "completion": result.completion,
            "rejection": result.rejection,
        }

    async def cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Cancel a held invocation under its authorization."""
        auth = RouteAuthorization.model_validate(payload["auth"])
        result = await self._deputy.cancel(str(payload["session_id"]), auth)
        return {
            "ok": result.ok,
            "cancelled": result.cancelled,
            "rejection": result.rejection,
        }

    async def stop(self) -> None:
        """Drop every bound sidecar on shutdown."""
        for replica_id in list(self._sidecars):
            await self.unbind_sidecar(replica_id)

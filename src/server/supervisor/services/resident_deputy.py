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

from ...network.state import ResolvedRoute, Transport
from ...resident.deputy import BootstrapResult, ResidentInvocationDeputy, StreamResult
from ...resident.relay_delivery import ResidentRelayEndpoint
from ...resident.sidecar import LoadEvidence, SidecarClaimGate
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
        endpoint: ResidentRelayEndpoint | None = None,
        relay_window_bytes: int = 65536,
        logger: logging.Logger | None = None,
    ) -> None:
        self._deputy = ResidentInvocationDeputy(
            connect_budget_sec=connect_budget_sec, logger=logger
        )
        self._endpoint = endpoint
        self._relay_sessions: set[str] = set()
        self._sidecars: dict[str, ResidentSidecarListener] = {}
        # Size the engine's per-chunk emission well under the relay window so a chunk
        # frame, with its wire envelope, always fits and never blocks the sender.
        self._engine_open = engine_open or HttpEngineDelivery(
            timeout_sec=engine_timeout_sec,
            chunk_chars=max(256, relay_window_bytes // 8),
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
            on_load=self._record_load,
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

    def _is_reverse_relay(self, route: ResolvedRoute) -> bool:
        """Whether the base candidate is the reverse-rendezvous control_relay."""
        return (
            self._endpoint is not None
            and bool(route.candidates)
            and route.candidates[0].transport is Transport.CONTROL_RELAY
        )

    async def bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Phase 1: deliver the bootstrap and report the enqueue acknowledgement."""
        route = ResolvedRoute.model_validate(payload["resolved_route"])
        handoff = AdmissionHandoff.model_validate(payload["handoff"])
        session_id = str(payload["session_id"])
        if self._is_reverse_relay(route):
            assert self._endpoint is not None
            self._relay_sessions.add(session_id)
            result: BootstrapResult = await self._endpoint.bootstrap(
                session_id,
                route=route,
                handoff=handoff,
                request_payload=payload.get("request"),
            )
        else:
            result = await self._deputy.bootstrap(
                session_id, route, handoff, payload.get("request")
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
        session_id = str(payload["session_id"])
        if session_id in self._relay_sessions:
            assert self._endpoint is not None
            result: StreamResult = await self._endpoint.stream(session_id, auth)
        else:
            result = await self._deputy.stream(session_id, auth)
        return {
            "ok": result.ok,
            "completion": result.completion,
            "rejection": result.rejection,
            "definite": result.definite,
        }

    async def cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reap a held invocation so both ends close and the engine request aborts."""
        session_id = str(payload["session_id"])
        if session_id in self._relay_sessions:
            assert self._endpoint is not None
            self._relay_sessions.discard(session_id)
            await self._endpoint.cancel(session_id)
            return {"ok": True, "cancelled": True, "rejection": None}
        result = await self._deputy.cancel(session_id)
        return {
            "ok": result.ok,
            "cancelled": result.cancelled,
            "rejection": result.rejection,
        }

    def _record_load(self, evidence: LoadEvidence) -> None:
        """Emit claim-tagged load evidence, tagged latency-sensitive versus bulk."""
        if self._logger is not None:
            self._logger.info(
                "resident load claim=%s invocation=%s replica=%s/%d op=%s class=%s",
                evidence.claim_id,
                evidence.invocation_id,
                evidence.replica_id,
                evidence.incarnation,
                evidence.operation,
                evidence.traffic_class.value,
            )

    async def stop(self) -> None:
        """Reap held deputy sessions and drop every bound sidecar on shutdown."""
        await self._deputy.aclose()
        for replica_id in list(self._sidecars):
            await self.unbind_sidecar(replica_id)

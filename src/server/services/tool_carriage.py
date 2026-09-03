"""The in-server origin of remote external-tool carriage.

``RemoteSidecarCarriage`` sits behind the ``ExecutionTransport`` seam: the broker hands
it an approved operation and it carries that operation to a remote worker sidecar over
the network plane, returning the typed outcome. It is claim-free — it mints no
``ServiceClaim``, ``RouteAuthorization``, or lease, and is not the resident relay. The
broker stays the sole authority; this only moves the operation and its result.

Three pieces cooperate: ``ToolTargetRegistry`` binds a control-issued sidecar on a
worker node and caches its ``NonresidentSidecarTarget``; ``ToolEgressOriginDeputy``
executes a resolved candidate ladder (a forward dial, or the tool reverse-rendezvous) to
deliver one operation frame and read its reply; and ``RemoteSidecarCarriage`` resolves
the route, builds the operation fence, and bridges the broker's synchronous call onto
the server main loop where the network I/O lives.
"""

import asyncio
import logging
import socket
import ssl
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from shared.schemas.command import CommandType
from shared.utils.ids import (
    new_route_origin_id,
    new_sidecar_target_id,
    new_tool_delivery_nonce,
    new_tool_relay_session_id,
)

from ..network import wire as netwire
from ..network.reachability import NetworkReachabilityView
from ..network.resolver import resolve_route
from ..network.state import (
    NetworkEndpointAdvertisement,
    NonresidentSidecarTarget,
    PolicyClass,
    ResolvedRoute,
    RouteCandidate,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
)
from ..orchestration.tool_dispatch import ToolOutcome, ToolOutcomeStatus
from . import tool_sidecar_wire as wire
from .tool_egress import (
    AmbiguousDelivery,
    CarriageResult,
    RemoteToolOperationEnvelope,
    ToolOperationEnvelope,
    ToolRequest,
    tool_request_digest,
)
from .tool_relay_delivery import ToolRelayEndpoint

# From the tool-carriage config edge: the target node to bind a sidecar on, and one node
# command exec against a node id. Both are injected so the carriage stays testable.
TargetNodeSelector = Callable[[], Awaitable[str | None]]
NodeEndpointProvider = Callable[[str], Awaitable[NetworkEndpointAdvertisement | None]]
ExecNodeCmd = Callable[[str, CommandType, dict[str, Any]], Awaitable[dict[str, Any]]]


def _unavailable(value: str) -> ToolOutcome:
    return ToolOutcome(status=ToolOutcomeStatus.UNAVAILABLE, value=value)


class ToolEgressOriginDeputy:
    """Executes a resolved candidate ladder to deliver one operation and read its reply.

    A forward-dial candidate is tried in order; a pre-send connect failure falls through
    to the next candidate, but once the operation is on the wire an ambiguous loss stops
    rather than re-driving to another path. ``control_relay`` is the guaranteed base,
    delivered over the tool reverse-rendezvous and never demoted.
    """

    def __init__(
        self,
        *,
        relay_endpoint: ToolRelayEndpoint,
        connect_budget_sec: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self._relay = relay_endpoint
        self._budget = connect_budget_sec
        self._log = logger

    async def deliver(
        self,
        session_id: str,
        route: ResolvedRoute,
        *,
        invocation_id: str,
        idm: str,
        operation_payload: bytes,
        read_deadline_sec: float,
    ) -> tuple[bytes | None, list[tuple[Transport, RouteObservationOutcome]], bool]:
        """Deliver one operation and read its reply.

        Returns ``(reply, observations, egressed)``. ``reply`` is the sidecar's opaque
        wire body or ``None`` when no terminal reply arrived. ``egressed`` is ``True``
        once the operation reached a wire — a forward-dial write completed, or a relay
        frame was published up — so a subsequent lost reply is an ambiguous delivery the
        caller holds pending, not a pre-delivery failure it may terminalize.
        """
        observations: list[tuple[Transport, RouteObservationOutcome]] = []
        for candidate in route.candidates:
            if candidate.transport is Transport.CONTROL_RELAY:
                origin_hop, target_hop = candidate.hops
                reply = await self._relay.deliver(
                    session_id,
                    invocation_id=invocation_id,
                    idm=idm,
                    origin_node=origin_hop.node_id or "",
                    target_node=target_hop.node_id or "",
                    sidecar_route=target_hop.endpoint,
                    operation_payload=operation_payload,
                    recv_timeout=read_deadline_sec,
                )
                # The frame was published up: a missing reply is an ambiguous loss.
                return reply, observations, True
            reply, outcome, ambiguous = await self._forward_dial(
                candidate, operation_payload, read_deadline_sec
            )
            observations.append((candidate.transport, outcome))
            if reply is not None:
                return reply, observations, False
            if ambiguous:
                return None, observations, True
        return None, observations, False

    async def cancel(self, session_id: str) -> None:
        """Best-effort reap of an in-flight delivery's origin/target/session state."""
        await self._relay.cancel(session_id)

    async def _forward_dial(
        self, candidate: RouteCandidate, payload: bytes, read_deadline_sec: float
    ) -> tuple[bytes | None, RouteObservationOutcome, bool]:
        # The connect is bounded by the connect budget; the reply read is bounded by the
        # operation deadline, since the sidecar answers only after its provider egress —
        # reading under the connect budget would time out a slow-but-healthy search.
        try:
            reader, writer = await asyncio.wait_for(
                self._connect(candidate), self._budget
            )
        except (TimeoutError, OSError, ValueError) as exc:
            return None, _classify(exc), False
        try:
            await netwire.write_frame(writer, payload)
            reply = await asyncio.wait_for(
                netwire.read_frame(reader), read_deadline_sec
            )
            return reply, RouteObservationOutcome.VERIFIED, False
        except (
            TimeoutError,
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            ValueError,
        ):
            # The operation is already on the wire: an ambiguous delivery, not a
            # pre-send failure, so stop rather than re-drive to another path. The
            # connect succeeded, so a slow or lost reply is application latency,
            # classified non-demoting so provider slowness never demotes the path.
            return None, RouteObservationOutcome.APPLICATION_ERROR, True
        finally:
            await _close(writer)

    async def _connect(
        self, candidate: RouteCandidate
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        host, port = wire.split_host_port(candidate.hops[0].endpoint)
        reader, writer = await asyncio.open_connection(host, port)
        for hop in candidate.hops[1:]:
            await netwire.write_frame(writer, hop.endpoint.encode())
        return reader, writer


class ToolTargetRegistry:
    """Binds a control-issued sidecar on a worker node and caches its target."""

    def __init__(
        self,
        *,
        exec_node_cmd: ExecNodeCmd,
        select_node: TargetNodeSelector,
        sidecar_route: str,
        provider: str,
        interfaces: tuple[str, ...],
        directly_routable: bool,
        policy_class: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:
        self._exec = exec_node_cmd
        self._select_node = select_node
        self._route = sidecar_route
        self._provider = provider
        self._interfaces = interfaces
        self._directly_routable = directly_routable
        self._policy_class = policy_class
        self._log = logger
        self._target: NonresidentSidecarTarget | None = None
        self._generation = 0
        self._lock = asyncio.Lock()

    async def ensure_target(self) -> NonresidentSidecarTarget | None:
        async with self._lock:
            if self._target is not None:
                return self._target
            node_id = await self._select_node()
            if node_id is None:
                return None
            self._generation += 1
            gen = self._generation
            target_id = new_sidecar_target_id()
            data = await self._exec(
                node_id,
                CommandType.BIND_TOOL_SIDECAR,
                {
                    "target_id": target_id,
                    "target_generation": gen,
                    "route": self._route,
                    "interfaces": list(self._interfaces),
                    "policy_class": self._policy_class,
                },
            )
            host, port = data.get("host"), data.get("port")
            if not host or not port:
                return None
            self._target = NonresidentSidecarTarget(
                target_id=target_id,
                target_generation=gen,
                node_id=node_id,
                incarnation=gen,
                listener_generation=gen,
                interfaces=self._interfaces,
                provider=self._provider,
                routes=(f"{host}:{port}",),
                directly_routable=self._directly_routable,
            )
            return self._target

    async def invalidate(self, target: NonresidentSidecarTarget) -> None:
        """Drop the cached target after a lost delivery so the next op rebinds.

        Only clears the still-current target, so a concurrent op that already rebound a
        fresh target is not reset; a bind bumps the generation, fencing the stale one.
        The old sidecar is unbound best-effort so a rebind against a live node leaves no
        orphaned listener behind.
        """
        async with self._lock:
            if self._target is None or self._target.target_id != target.target_id:
                return
            self._target = None
        try:
            await self._exec(
                target.node_id,
                CommandType.UNBIND_TOOL_SIDECAR,
                {"target_id": target.target_id},
            )
        except Exception as exc:  # noqa: BLE001 - unbind is best-effort
            if self._log is not None:
                self._log.debug("stale tool sidecar unbind failed: %s", exc)

    async def close(self) -> None:
        target = self._target
        self._target = None
        if target is None:
            return
        try:
            await self._exec(
                target.node_id,
                CommandType.UNBIND_TOOL_SIDECAR,
                {"target_id": target.target_id},
            )
        except Exception as exc:  # noqa: BLE001 - unbind is best-effort on shutdown
            if self._log is not None:
                self._log.debug("tool sidecar unbind failed: %s", exc)


class RemoteSidecarCarriage:
    """Carries an approved operation to a remote worker sidecar off-server."""

    def __init__(
        self,
        *,
        origin_deputy: ToolEgressOriginDeputy,
        registry: ToolTargetRegistry,
        endpoint_provider: NodeEndpointProvider,
        ingress_endpoint: NetworkEndpointAdvertisement,
        provider: str,
        tenant: str | None = None,
        policy_class: str = "default",
        deadline_sec: float = 30.0,
        route_ttl_sec: float = 30.0,
        connect_budget_sec: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._deputy = origin_deputy
        self._registry = registry
        self._endpoint_provider = endpoint_provider
        self._ingress = ingress_endpoint
        self._provider = provider
        self._tenant = tenant
        self._policy_class = policy_class
        self._deadline_sec = deadline_sec
        self._route_ttl = route_ttl_sec
        self._budget = connect_budget_sec
        self._log = logger or logging.getLogger("remote-sidecar-carriage")
        self._reach = NetworkReachabilityView()
        self._origin_id = new_route_origin_id()
        self._epoch = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the server main loop the network I/O runs on (at lifespan startup)."""
        self._loop = loop

    def __call__(
        self, envelope: ToolOperationEnvelope, request: ToolRequest
    ) -> CarriageResult:
        loop = self._loop
        if loop is None:
            return _unavailable("the remote tool carriage is not ready")
        session_id = new_tool_relay_session_id()
        future = asyncio.run_coroutine_threadsafe(
            self._deliver(envelope, request, session_id), loop
        )
        try:
            # Cover a connect on each of up to three serial candidates plus one full
            # provider egress, so a slow search reached only after a failed forward dial
            # is not cut off by the outer bound.
            outer = envelope.timeout_sec + self._budget * 3 + 10.0
            return future.result(timeout=outer)
        except FuturesTimeoutError:
            # The operation outran its bound after it was on a wire: reap the in-flight
            # session on both ends and hold the boundary pending. It is never coerced to
            # a terminal outcome, since the sidecar may have egressed; a re-drive reuses
            # the same idm-* under a fresh nonce and session.
            self._abort(session_id)
            future.cancel()
            return AmbiguousDelivery("the remote tool operation outran its bound")
        except Exception as exc:  # noqa: BLE001 - a carriage fault is a typed outcome
            self._log.warning("remote tool carriage failed: %s", exc)
            return _unavailable("the remote tool carriage failed")

    def _abort(self, session_id: str) -> None:
        loop = self._loop
        if loop is None:
            return
        # Fire the cancel poke on the loop the network I/O runs on and wait briefly, so
        # a reverse-rendezvous target reaps its runner promptly rather than only on its
        # own bounded read; a reap failure is harmless best-effort.
        try:
            asyncio.run_coroutine_threadsafe(
                self._deputy.cancel(session_id), loop
            ).result(timeout=self._budget)
        except Exception as exc:  # noqa: BLE001 - cancel/reap is best-effort
            self._log.debug("remote tool cancel failed session=%s: %s", session_id, exc)

    async def _deliver(
        self, envelope: ToolOperationEnvelope, request: ToolRequest, session_id: str
    ) -> CarriageResult:
        target = await self._registry.ensure_target()
        if target is None or envelope.interface not in target.interfaces:
            return _unavailable("no eligible external-tool sidecar target")
        target_endpoint = await self._endpoint_provider(target.node_id)
        origin = self._route_origin()
        now = time.monotonic()
        self._epoch += 1
        route = resolve_route(
            origin,
            target,
            target_endpoint,
            self._reach,
            now=now,
            route_epoch=self._epoch,
            expires_at=now + self._route_ttl,
        )
        if not route.candidates:
            return _unavailable("no route to the external-tool sidecar")
        for candidate in route.candidates:
            self._reach.mark_optimistic(
                origin.origin_id,
                origin.policy_class,
                target.node_id,
                target.incarnation,
                target.listener_generation,
                candidate.transport,
                now=now,
            )
        payload = wire.encode_msg(
            wire.KIND_OPERATION,
            envelope=self._fence(envelope, request, target).model_dump(mode="json"),
            request=request.model_dump(mode="json"),
        )
        idm = envelope.idempotency_key or session_id
        reply, observations, egressed = await self._deputy.deliver(
            session_id,
            route,
            invocation_id=idm,
            idm=idm,
            operation_payload=payload,
            read_deadline_sec=envelope.timeout_sec + self._budget,
        )
        self._record(origin, target, observations)
        if reply is not None:
            return self._decode(reply)
        # No terminal reply: drop the cached target so the next physical attempt rebinds
        # a fresh sidecar under a rotated generation rather than resolving forever to a
        # target that may be gone.
        await self._registry.invalidate(target)
        if egressed:
            # The operation reached a wire and may have egressed: hold the durable
            # boundary pending for a same-idm-* re-drive, never a terminal outcome.
            return AmbiguousDelivery("the remote tool reply was lost after egress")
        # The operation never reached a wire — a known pre-delivery failure, safe to
        # terminalize since nothing egressed.
        return _unavailable("no route to the external-tool sidecar")

    def _fence(
        self,
        envelope: ToolOperationEnvelope,
        request: ToolRequest,
        target: NonresidentSidecarTarget,
    ) -> RemoteToolOperationEnvelope:
        return RemoteToolOperationEnvelope(
            interface=envelope.interface,
            provider=self._provider,
            idempotency_key=envelope.idempotency_key,
            request_digest=tool_request_digest(
                request.interface, request.query, request.max_results
            ),
            target_id=target.target_id,
            target_generation=target.target_generation,
            delivery_nonce=new_tool_delivery_nonce(),
            tenant=self._tenant,
            policy_class=self._policy_class,
            deadline_epoch=time.time() + self._deadline_sec,
            max_results=envelope.max_results,
            timeout_sec=envelope.timeout_sec,
            result_char_cap=envelope.result_char_cap,
        )

    def _record(
        self,
        origin: RouteOrigin,
        target: NonresidentSidecarTarget,
        observations: list[tuple[Transport, RouteObservationOutcome]],
    ) -> None:
        now = time.monotonic()
        for transport, outcome in observations:
            self._reach.observe(
                RouteObservation(
                    origin_id=origin.origin_id,
                    policy_class=origin.policy_class,
                    target_node_id=target.node_id,
                    incarnation=target.incarnation,
                    listener_generation=target.listener_generation,
                    transport=transport,
                    outcome=outcome,
                ),
                now=now,
            )

    def _decode(self, reply: bytes | None) -> ToolOutcome:
        if reply is None:
            return _unavailable("the remote tool operation was lost")
        try:
            msg = wire.decode_msg(reply)
        except ValueError:
            return _unavailable("the remote tool reply was malformed")
        if msg["kind"] == wire.KIND_RESULT:
            return ToolOutcome.model_validate(msg["outcome"])
        if msg["kind"] == wire.KIND_REJECT:
            return _unavailable(
                f"the remote sidecar rejected the operation: {msg.get('reason')}"
            )
        return _unavailable("the remote tool reply was unexpected")

    def _route_origin(self) -> RouteOrigin:
        return RouteOrigin(
            origin_id=self._origin_id,
            endpoint_id=self._ingress.endpoint_id,
            node_id=self._ingress.node_id,
            reachability_class=self._ingress.reachability_class,
            policy_class=PolicyClass.DEFAULT,
            trust_domain=self._ingress.trust_domain,
            relay_attachment_id=self._ingress.relay_attachment_id,
        )


def _classify(exc: BaseException) -> RouteObservationOutcome:
    if isinstance(exc, TimeoutError):
        return RouteObservationOutcome.TIMEOUT
    if isinstance(exc, ConnectionRefusedError):
        return RouteObservationOutcome.CONNECT_FAILURE
    if isinstance(exc, ssl.SSLError):
        return RouteObservationOutcome.TLS_FAILURE
    if isinstance(exc, socket.gaierror):
        return RouteObservationOutcome.DNS_FAILURE
    return RouteObservationOutcome.ROUTE_FAILURE


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass


__all__ = [
    "ExecNodeCmd",
    "NodeEndpointProvider",
    "RemoteSidecarCarriage",
    "TargetNodeSelector",
    "ToolEgressOriginDeputy",
    "ToolTargetRegistry",
]

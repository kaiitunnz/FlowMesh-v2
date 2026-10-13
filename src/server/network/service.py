"""The network-plane control service.

Ties the derived reachability view to the node registry (the durable carrier of endpoint
advertisements) and the pure resolver. It resolves an ordered route for a trusted
origin, records the deputy's classified observations, and exposes endpoints and
reachability for diagnostics.
"""

import asyncio
import logging
import time
from collections.abc import Mapping

from shared.network.frame_stream import split_host_port
from shared.schemas.network import RESIDENT_PEER_PROTOCOL
from shared.utils.ids import new_route_origin_id

from ..config import NetworkPlaneConfig
from ..registries.node import NodeRegistry
from .reachability import NetworkReachabilityView, ReachabilityBounds
from .resolver import resolve_route
from .state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    ResolvedRoute,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    RouteTarget,
    Transport,
    TrustedPeerPolicy,
)


def stamp_endpoint(
    endpoint: NetworkEndpointAdvertisement,
    node_id: str,
    *,
    attachment_prefix: str = "rr",
) -> NetworkEndpointAdvertisement:
    """Stamp a raw advertisement's node id and complete an outbound-only identity.

    A network-plane node always attaches outward to the rendezvous, so it is relay
    attach-eligible even with no inbound URL. Its endpoint and attachment identities
    derive from the node id when the advertisement carries none; ``attachment_prefix``
    selects the namespace whose outbound attachment the presence signals (``rr``
    resident, ``xt`` tool).
    """
    updates: dict[str, str] = {"node_id": node_id}
    if not endpoint.endpoint_id:
        updates["endpoint_id"] = f"node-{node_id}"
    if endpoint.relay_attachment_id is None:
        updates["relay_attachment_id"] = f"{attachment_prefix}-{node_id}"
    return endpoint.model_copy(update=updates)


_REACHABILITY_CLASSES = frozenset(ReachabilityClass)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _peer_policy(config: NetworkPlaneConfig) -> TrustedPeerPolicy:
    """The pure resolver's trust rule, derived from the deployment's peer posture."""
    peer = config.peer
    return TrustedPeerPolicy(
        enabled=peer.enabled,
        trust_domain=peer.trust_domain,
        classes=frozenset(
            ReachabilityClass(value)
            for value in peer.classes
            if value in _REACHABILITY_CLASSES
        ),
        protocol=RESIDENT_PEER_PROTOCOL,
    )


class NetworkPlane:
    """Control-plane route discovery over trusted endpoint advertisements."""

    def __init__(
        self,
        config: NetworkPlaneConfig,
        node_registry: NodeRegistry,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._nodes = node_registry
        self._logger = logger
        self._reachability = NetworkReachabilityView(
            ReachabilityBounds(
                positive_ttl_sec=config.positive_ttl_sec,
                negative_ttl_sec=config.negative_ttl_sec,
                backoff_base_sec=config.backoff_base_sec,
                backoff_max_sec=config.backoff_max_sec,
            )
        )
        self._route_epoch = 0
        self._origin_ids: dict[tuple[str, PolicyClass, int], str] = {}
        self._seen_generation: dict[str, int] = {}
        self._trust = _peer_policy(config)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Keep the view on ``loop``, the server's own, whatever thread reports a
        node's departure."""
        self._loop = loop

    @property
    def connect_budget_sec(self) -> float:
        return self._config.connect_budget_sec

    @property
    def route_ttl_sec(self) -> float:
        return self._config.route_ttl_sec

    @property
    def worker_directly_routable(self) -> bool:
        """Whether the deployment declares its workers' listeners reachable off-host."""
        return self._config.worker_directly_routable

    def peer_trust(self, protocol: str) -> TrustedPeerPolicy:
        """Return the deployment's peer rule for a route carrying ``protocol``."""
        return self._trust.model_copy(update={"protocol": protocol})

    @property
    def probe_trust(self) -> TrustedPeerPolicy:
        """The rule a reachability probe resolves under.

        Where the deployment serves peer listeners, a probe is offered every reachable
        peer path regardless of the pair's trust, since the listener answers it without
        any session; where it serves none, a probe is offered no peer path. A route
        carrying resident traffic never resolves under this rule.
        """
        if not self._config.peer.enabled:
            return TrustedPeerPolicy()
        return TrustedPeerPolicy(enabled=True, probe=True)

    async def endpoint_for(self, node_id: str) -> NetworkEndpointAdvertisement | None:
        """The node's current advertisement, stamped with its assigned node id."""
        node = await self._nodes.get_node_async(node_id)
        if node is None or node.network_endpoint is None:
            return None
        return stamp_endpoint(node.network_endpoint, node.id)

    @staticmethod
    def node_host(endpoint: NetworkEndpointAdvertisement | None) -> str:
        """Return the host a node is reached at; empty when it advertises none."""
        if endpoint is None or not endpoint.url:
            return ""
        return split_host_port(endpoint.url)[0]

    @staticmethod
    def listener_address(
        endpoint: NetworkEndpointAdvertisement | None, port: int
    ) -> str:
        """Return the ``host:port`` a listener a worker bound on ``port`` is dialed at,
        on its node's host; empty when it binds none or the node advertises no host."""
        host = NetworkPlane.node_host(endpoint)
        return f"{host}:{port}" if host and port > 0 else ""

    async def _endpoint(
        self,
        node_id: str,
        known: Mapping[str, NetworkEndpointAdvertisement | None],
    ) -> NetworkEndpointAdvertisement | None:
        if node_id in known:
            return known[node_id]
        return await self.endpoint_for(node_id)

    async def resolve(
        self,
        origin_node_id: str,
        listener: RouteTarget,
        *,
        trust: TrustedPeerPolicy | None = None,
        policy_class: PolicyClass = PolicyClass.DEFAULT,
        endpoints: Mapping[str, NetworkEndpointAdvertisement | None] | None = None,
    ) -> tuple[RouteOrigin, ResolvedRoute] | None:
        """Resolve an ordered candidate ladder from origin to the target listener.

        The origin node is the one whose deputy will execute the route, so it is what
        every candidate — the peer transports included — is graded against. ``trust``
        overrides the deployment policy for a caller that probes reachability, or for
        an origin restricted to ``control_relay``. ``policy_class`` scopes the origin
        and its evidence among the dialers sharing that node's endpoint. ``endpoints``
        are advertisements the caller already read, by node id, so neither end is read
        again.

        Returns ``None`` when the origin advertises no network endpoint.
        """
        known = endpoints or {}
        origin_endpoint = await self._endpoint(origin_node_id, known)
        if origin_endpoint is None:
            return None
        target_endpoint = await self._endpoint(listener.node_id, known)
        self._invalidate_on_rotation(target_endpoint)

        origin = self._route_origin(origin_endpoint, origin_node_id, policy_class)
        now = time.monotonic()
        self._route_epoch += 1
        route = resolve_route(
            origin,
            listener,
            target_endpoint,
            self._reachability,
            trust=self._trust if trust is None else trust,
            now=now,
            route_epoch=self._route_epoch,
            expires_at=now + self._config.route_ttl_sec,
        )
        for candidate in route.candidates:
            self._reachability.mark_optimistic(
                origin.origin_id,
                origin.policy_class,
                listener.node_id,
                listener.incarnation,
                listener.listener_generation,
                candidate.transport,
                now=now,
            )
        return origin, route

    def record_observations(
        self,
        origin: RouteOrigin,
        listener: RouteTarget,
        observations: list[tuple[Transport, RouteObservationOutcome]],
    ) -> None:
        """Fold the deputy's classified observations into the reachability view."""
        now = time.monotonic()
        for transport, outcome in observations:
            self._reachability.observe(
                RouteObservation(
                    origin_id=origin.origin_id,
                    policy_class=origin.policy_class,
                    target_node_id=listener.node_id,
                    incarnation=listener.incarnation,
                    listener_generation=listener.listener_generation,
                    transport=transport,
                    outcome=outcome,
                ),
                now=now,
            )

    def reachability_states(
        self, origin: RouteOrigin, listener: RouteTarget
    ) -> dict[str, str]:
        """The current directional state per transport, for the echo response."""
        now = time.monotonic()
        return {
            transport.value: self._reachability.state_for(
                origin.origin_id,
                origin.policy_class,
                listener.node_id,
                listener.incarnation,
                listener.listener_generation,
                transport,
                now=now,
            ).value
            for transport in Transport
        }

    def reachability_snapshot(self) -> list[dict[str, str | int]]:
        now = time.monotonic()
        snapshot: list[dict[str, str | int]] = []
        for entry in self._reachability.entries():
            state = self._reachability.state_for(
                entry.origin_id,
                entry.policy_class,
                entry.target_node_id,
                entry.incarnation,
                entry.listener_generation,
                entry.transport,
                now=now,
            )
            snapshot.append(
                {
                    "origin_id": entry.origin_id,
                    "policy_class": entry.policy_class.value,
                    "target_node_id": entry.target_node_id,
                    "incarnation": entry.incarnation,
                    "listener_generation": entry.listener_generation,
                    "transport": entry.transport.value,
                    "state": state.value,
                    "retries": entry.retries,
                }
            )
        return snapshot

    def forget_node(self, node_id: str) -> None:
        """Reclaim a departed node's reachability entries and rotation cursor.

        Wired to node deregistration so the derived view stays bounded under node churn.
        """
        loop = self._loop
        if loop is None or _running_loop() is loop:
            self._forget_node(node_id)
        else:
            loop.call_soon_threadsafe(self._forget_node, node_id)

    def _forget_node(self, node_id: str) -> None:
        self._reachability.invalidate_node(node_id)
        self._seen_generation.pop(node_id, None)

    async def endpoints(self) -> list[NetworkEndpointAdvertisement]:
        nodes = await self._nodes.list_nodes_async()
        return [
            stamp_endpoint(node.network_endpoint, node.id)
            for node in nodes
            if node.network_endpoint is not None
        ]

    def _route_origin(
        self,
        endpoint: NetworkEndpointAdvertisement,
        node_id: str,
        policy_class: PolicyClass,
    ) -> RouteOrigin:
        # A fresh generation re-binds the origin to a new id, orphaning its prior route
        # memory; a stable endpoint keeps one id so reachability accumulates over calls.
        key = (endpoint.endpoint_id, policy_class, endpoint.generation)
        origin_id = self._origin_ids.get(key)
        if origin_id is None:
            origin_id = new_route_origin_id()
            self._origin_ids[key] = origin_id
        return RouteOrigin(
            origin_id=origin_id,
            endpoint_id=endpoint.endpoint_id,
            node_id=node_id,
            reachability_class=endpoint.reachability_class,
            policy_class=policy_class,
            trust_domain=endpoint.trust_domain,
            protocols=endpoint.protocols,
            relay_attachment_id=endpoint.relay_attachment_id,
        )

    def _invalidate_on_rotation(
        self, target_endpoint: NetworkEndpointAdvertisement | None
    ) -> None:
        if target_endpoint is None or target_endpoint.node_id is None:
            return
        node_id = target_endpoint.node_id
        seen = self._seen_generation.get(node_id)
        if seen is not None and target_endpoint.generation > seen:
            self._reachability.invalidate_node(node_id)
        self._seen_generation[node_id] = target_endpoint.generation

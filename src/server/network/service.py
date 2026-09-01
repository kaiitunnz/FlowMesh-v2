"""The network-plane control service.

Ties the derived reachability view to the node registry (the durable carrier of endpoint
advertisements) and the pure resolver. It resolves an ordered route for a trusted
origin, records the deputy's classified observations, and exposes the directory and
reachability for diagnostics. It admits no capacity and issues no route authorization.
"""

import logging
import time

from shared.utils.ids import new_route_origin_id

from ..config import NetworkPlaneConfig
from ..registries.node import NodeRegistry
from .reachability import NetworkReachabilityView, ReachabilityBounds
from .resolver import resolve_route
from .state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteObservation,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
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

    @property
    def connect_budget_sec(self) -> float:
        return self._config.connect_budget_sec

    async def endpoint_for(self, node_id: str) -> NetworkEndpointAdvertisement | None:
        """The node's current advertisement, stamped with its assigned node id."""
        node = await self._nodes.get_node_async(node_id)
        if node is None or node.network_endpoint is None:
            return None
        return node.network_endpoint.model_copy(update={"node_id": node.id})

    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute] | None:
        """Resolve an ordered candidate ladder from origin to the target listener.

        Returns ``None`` when the origin advertises no network endpoint.
        """
        origin_endpoint = await self.endpoint_for(origin_node_id)
        if origin_endpoint is None:
            return None
        target_endpoint = await self.endpoint_for(listener.node_id)
        self._invalidate_on_rotation(target_endpoint)

        origin = self._route_origin(origin_endpoint, origin_node_id)
        now = time.monotonic()
        self._route_epoch += 1
        route = resolve_route(
            origin,
            listener,
            target_endpoint,
            self._reachability,
            now=now,
            route_epoch=self._route_epoch,
            control_relay_endpoint=self._config.control_relay_url,
            expires_at=now + self._config.route_ttl_sec,
        )
        for candidate in route.candidates:
            self._reachability.mark_optimistic(
                origin.origin_id,
                origin.policy_class,
                listener.node_id,
                listener.listener_generation,
                candidate.transport,
                now=now,
            )
        return origin, route

    def record_observations(
        self,
        origin: RouteOrigin,
        listener: ReplicaListenerAdvertisement,
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
                    listener_generation=listener.listener_generation,
                    transport=transport,
                    outcome=outcome,
                ),
                now=now,
            )

    def reachability_states(
        self, origin: RouteOrigin, listener: ReplicaListenerAdvertisement
    ) -> dict[str, str]:
        """The current directional state per transport, for the echo response."""
        now = time.monotonic()
        return {
            transport.value: self._reachability.state_for(
                origin.origin_id,
                origin.policy_class,
                listener.node_id,
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
                entry.listener_generation,
                entry.transport,
                now=now,
            )
            snapshot.append(
                {
                    "origin_id": entry.origin_id,
                    "target_node_id": entry.target_node_id,
                    "listener_generation": entry.listener_generation,
                    "transport": entry.transport.value,
                    "state": state.value,
                }
            )
        return snapshot

    async def endpoints(self) -> list[NetworkEndpointAdvertisement]:
        nodes = await self._nodes.list_nodes_async()
        return [
            node.network_endpoint.model_copy(update={"node_id": node.id})
            for node in nodes
            if node.network_endpoint is not None
        ]

    def _route_origin(
        self, endpoint: NetworkEndpointAdvertisement, node_id: str
    ) -> RouteOrigin:
        # A fresh generation re-binds the origin to a new id, orphaning its prior route
        # memory; a stable endpoint keeps one id so reachability accumulates over calls.
        key = (endpoint.endpoint_id, PolicyClass.DEFAULT, endpoint.generation)
        origin_id = self._origin_ids.get(key)
        if origin_id is None:
            origin_id = new_route_origin_id()
            self._origin_ids[key] = origin_id
        return RouteOrigin(
            origin_id=origin_id,
            endpoint_id=endpoint.endpoint_id,
            node_id=node_id,
            reachability_class=endpoint.reachability_class,
            policy_class=PolicyClass.DEFAULT,
            trust_domain=endpoint.trust_domain,
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

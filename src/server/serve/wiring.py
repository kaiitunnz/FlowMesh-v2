"""Assemble the gated serve subsystem from the server runtime and configuration.

Builds the binding, terminal, and exposure stores (restored from the persisted serve
snapshot), the origin relay, the ``GatedServe`` edge, and — when forward serve is
enabled — the root forward listener, returning the handles the lifespan still drives.
The deferred ``set_advertise_route`` seam is wired by the caller once the event monitor
exists, so it stays out of here.
"""

import logging
from dataclasses import dataclass

from ..config import PortForwardConfig
from ..network.reverse_relay import BinaryRedis
from ..registries.resident import ResidentRegistry
from ..resident.service import ResidentCapacityControl
from ..resident.target_leg import TargetLegSupport
from .binding import ServeBindingStore, ServeSnapshot
from .forward_exposure import ForwardIngressDirectory
from .forward_listener import RootForwardIngress
from .ingress import ServeIngressRegistry
from .relay import SERVE_EDGE_STREAM_ID, ServeRelayExecutor
from .service import GatedServe
from .state import ServeTerminalStore


@dataclass(frozen=True)
class GatedServeWiring:
    """The wired serve handles the lifespan still drives after construction."""

    gated_serve: GatedServe
    forward_ingress: RootForwardIngress | None
    bindings: ServeBindingStore


def build_forward_serve_ingress(
    config: PortForwardConfig,
    gated_serve: GatedServe,
    logger: logging.Logger,
) -> RootForwardIngress | None:
    """Build the root forward serve listener and attach it to the gated edge.

    Returns ``None`` when forward serve is disabled: the caller wires no listener and
    every forward-pinned request fails closed. The listener binds on the same interface
    and advertises the same public host SSH port forwarding uses.
    """
    if not config.serve_forward_enabled:
        return None
    listener = RootForwardIngress(
        bind_host=config.bind_host,
        public_host=config.public_host,
        admit=gated_serve.admit_forward_request,
        on_bound=gated_serve.commit_forward,
        body_budget_bytes=config.serve_forward_body_budget_bytes,
        logger=logger,
    )
    gated_serve.set_forward_listener(listener)
    return listener


def build_gated_serve(
    *,
    control: ResidentCapacityControl,
    registry: ResidentRegistry,
    relay_redis: BinaryRedis,
    port_forward: PortForwardConfig,
    target_leg: TargetLegSupport | None = None,
    logger: logging.Logger,
) -> GatedServeWiring:
    """Wire and return the gated serve subsystem.

    Restores the binding, terminal, and exposure stores from the persisted serve
    snapshot; a snapshot loaded with forward disabled retires its exposures so the
    directory never holds a live entry no listener backs. The persist closure captures
    the registry and the stores so every store mutation snapshots the same set.
    """
    bindings = ServeBindingStore()
    terminals = ServeTerminalStore()
    exposures = ForwardIngressDirectory(
        port_forward.public_host,
        port_forward.serve_forward_port_start,
        port_forward.serve_forward_port_end,
    )
    if (stored := registry.load_serve_snapshot()) is not None:
        bindings.load_snapshot(stored.bindings)
        terminals.load_snapshot(stored.terminals)
        exposures.load_snapshot(stored.exposures)
        if not port_forward.serve_forward_enabled:
            exposures.retire_all()

    def _persist() -> None:
        registry.save_serve_snapshot(
            ServeSnapshot(
                bindings=bindings.to_snapshot(),
                terminals=terminals.to_snapshot(),
                exposures=exposures.to_snapshot(),
            )
        )

    relay = ServeRelayExecutor(
        relay_redis=relay_redis,
        edge_id=SERVE_EDGE_STREAM_ID,
        control=control,
        target_leg=target_leg,
        logger=logger,
    )
    gated_serve = GatedServe(
        bindings=bindings,
        terminals=terminals,
        control=control,
        relay=relay,
        # An operator that refuses public serve exposure registers no root-local proxy
        # ingress, so every proxy-pinned request fails closed; a forward ingress is
        # registered only where a deployment hosts one.
        ingresses=ServeIngressRegistry(
            SERVE_EDGE_STREAM_ID if port_forward.serve_proxy_enabled else None
        ),
        # Each forward binding owns a per-task public port on the root's public host;
        # the root binds a plain-HTTP listener on it, and a task without a configured
        # forward public host/range fails closed. The directory persists so a restart
        # rebinds each live exposure to its same port.
        exposures=exposures,
        persist=_persist,
        logger=logger,
    )
    forward_ingress = build_forward_serve_ingress(port_forward, gated_serve, logger)
    return GatedServeWiring(
        gated_serve=gated_serve, forward_ingress=forward_ingress, bindings=bindings
    )

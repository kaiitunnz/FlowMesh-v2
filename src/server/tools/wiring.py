"""Builds the remote external-tool carriage from server configuration.

Assembles the claim-free tool relay bridge, the in-server ingress edge, the target
registry, and the carriage when the deployment routes external-tool egress to a remote
worker sidecar. Returns ``None`` when that path is not configured.
"""

import os
from dataclasses import dataclass
from logging import Logger

from shared.schemas.command import CommandMessage, CommandType
from shared.schemas.network import NetworkEndpointAdvertisement, ReachabilityClass

from ..config import ServerConfig
from ..network.rendezvous import RootCursorStore, RootRendezvousBridge
from ..network.reverse_relay import (
    TOOL_RELAY_KEYSPACE,
    BinaryRedis,
    RelaySessionStore,
    RelayStreamStore,
)
from ..network.service import stamp_endpoint
from ..registries import WorkerRegistry
from ..registries.node import NodeRegistry
from ..supervisor.services.reverse_relay_attachment import ReverseRelayAttachment
from ..task.runtime import TaskRuntime
from .tool_carriage import (
    RemoteSidecarCarriage,
    ToolEgressOriginDeputy,
    ToolTargetRegistry,
)
from .tool_egress import EgressLocality
from .tool_relay_delivery import ToolRelayEndpoint


@dataclass(frozen=True)
class RemoteToolCarriage:
    """The wired tool-carriage objects the server lifespan starts and stops."""

    bridge: RootRendezvousBridge
    ingress_attach: ReverseRelayAttachment
    ingress_node_id: str
    target_registry: ToolTargetRegistry
    carriage: RemoteSidecarCarriage


def _reachability_class(value: str) -> ReachabilityClass:
    try:
        return ReachabilityClass(value)
    except ValueError:
        return ReachabilityClass.ROUTABLE


def build_remote_tool_carriage(
    *,
    config: ServerConfig,
    node_registry: NodeRegistry,
    worker_registry: WorkerRegistry | None,
    runtime: TaskRuntime | None,
    relay_redis: BinaryRedis | None,
    logger: Logger,
) -> RemoteToolCarriage | None:
    ws = config.orchestration.web_search
    net = config.orchestration.network
    if not (
        net.enabled
        and ws.sidecar_remote
        and ws.egress_locality == EgressLocality.WORKER_SIDECAR.value
    ):
        return None
    if relay_redis is None:
        return None

    bridge = RootRendezvousBridge(
        RelayStreamStore(relay_redis, TOOL_RELAY_KEYSPACE),
        RelaySessionStore(relay_redis, TOOL_RELAY_KEYSPACE),
        RootCursorStore(relay_redis, TOOL_RELAY_KEYSPACE),
        logger=logger,
    )
    ingress_node_id = f"xt-ingress-{config.identity.alias or 'root'}"
    origin_endpoint = ToolRelayEndpoint(relay_redis, ingress_node_id, logger=logger)
    ingress_attach = ReverseRelayAttachment(
        relay_redis,
        ingress_node_id,
        origin_endpoint,
        owner=f"{ingress_node_id}:{os.getpid()}",
        keyspace=TOOL_RELAY_KEYSPACE,
        logger=logger,
    )
    ingress_endpoint = NetworkEndpointAdvertisement(
        endpoint_id=ingress_node_id,
        node_id=ingress_node_id,
        url="",
        generation=1,
        trust_domain=net.trust_domain,
        reachability_class=_reachability_class(net.reachability_class),
        relay_attachment_id=f"xt-{ingress_node_id}",
    )

    async def _exec_node_cmd(
        node_id: str, command: CommandType, payload: dict[str, object]
    ) -> dict[str, object]:
        resp = await node_registry.exec_node_cmd(
            node_id,
            CommandMessage(command=command, payload=payload),
            timeout=net.connect_budget_sec + ws.timeout_sec + 10.0,
        )
        if not resp.success:
            raise RuntimeError(resp.message or "tool node command failed")
        return resp.data or {}

    async def _resolve_target(task_id: str) -> tuple[str, str, int] | None:
        # The blocked Agent episode's assigned worker owns the activation's context;
        # route the operation to it and fence on its registration incarnation.
        record = runtime.get_record(task_id) if runtime is not None else None
        worker_id = record.assigned_worker if record is not None else None
        if not worker_id or worker_registry is None:
            return None
        worker = await worker_registry.get_worker_async(worker_id)
        if worker is None:
            return None
        return worker.node_id, worker.id, worker.incarnation

    async def _target_endpoint(node_id: str) -> NetworkEndpointAdvertisement | None:
        node = await node_registry.get_node_async(node_id)
        if node is None or node.network_endpoint is None:
            return None
        return stamp_endpoint(node.network_endpoint, node.id, attachment_prefix="xt")

    target_registry = ToolTargetRegistry(
        exec_node_cmd=_exec_node_cmd,
        resolve_target=_resolve_target,
        sidecar_route=ws.sidecar_route,
        provider=ws.provider,
        interfaces=("search/v1",),
        directly_routable=ws.sidecar_directly_routable,
        logger=logger,
    )
    carriage = RemoteSidecarCarriage(
        origin_deputy=ToolEgressOriginDeputy(
            relay_endpoint=origin_endpoint,
            connect_budget_sec=net.connect_budget_sec,
            logger=logger,
        ),
        registry=target_registry,
        endpoint_provider=_target_endpoint,
        ingress_endpoint=ingress_endpoint,
        provider=ws.provider,
        deadline_sec=ws.timeout_sec + 10.0,
        route_ttl_sec=net.route_ttl_sec,
        connect_budget_sec=net.connect_budget_sec,
        logger=logger,
    )
    return RemoteToolCarriage(
        bridge=bridge,
        ingress_attach=ingress_attach,
        ingress_node_id=ingress_node_id,
        target_registry=target_registry,
        carriage=carriage,
    )


__all__ = ["RemoteToolCarriage", "build_remote_tool_carriage"]

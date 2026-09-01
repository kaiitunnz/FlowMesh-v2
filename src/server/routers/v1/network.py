"""Feature-gated network-plane route-discovery diagnostics and echo seam.

The echo resolves a route from advertisements and reachability, delivers the plan to the
origin node's deputy, and folds the returned observations into the reachability view.
Every endpoint is SYSTEM/ADMIN.
"""

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from shared.schemas.command import CommandMessage, CommandType

from ...app_state import get_logger, get_network_plane, get_node_registry
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...network.service import NetworkPlane
from ...network.state import (
    ReplicaListenerAdvertisement,
    RouteObservationOutcome,
    Transport,
)
from ...network.wire import APP_ERROR_SENTINEL
from ...registries.node import NodeRegistry
from ...schemas.network import (
    NetworkEchoRequest,
    NetworkEchoResponse,
    NetworkEndpointInfo,
    NetworkReachabilityEntryInfo,
)

router = APIRouter(prefix="/network", tags=["Network"])

_ECHO_TIMEOUT_SEC = 30.0


async def _require_admin(principal: PrincipalContext, logger: logging.Logger) -> None:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )


def _require_plane(plane: NetworkPlane | None) -> NetworkPlane:
    if plane is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="network plane disabled"
        )
    return plane


@router.post(
    "/echo",
    summary="Resolve a route and echo over the selected transport",
    description="Resolve an ordered route to the target listener and round-trip a "
    "payload over the first working transport, updating reachability.",
)
async def network_echo(
    body: NetworkEchoRequest,
    principal: PrincipalContext = Depends(authenticate_connection),
    plane: NetworkPlane | None = Depends(get_network_plane),
    node_registry: NodeRegistry = Depends(get_node_registry),
    logger: logging.Logger = Depends(get_logger),
) -> NetworkEchoResponse:
    await _require_admin(principal, logger)
    network = _require_plane(plane)

    listener = ReplicaListenerAdvertisement(
        replica_id=body.listener.replica_id,
        family=body.listener.family,
        incarnation=body.listener.incarnation,
        listener_generation=body.listener.listener_generation,
        node_id=body.listener.node_id,
        routes=tuple(body.listener.routes),
        directly_routable=body.listener.directly_routable,
    )
    resolved = await network.resolve(body.origin_node_id, listener)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="origin node has no network endpoint advertisement",
        )
    origin, route = resolved

    echo_payload = APP_ERROR_SENTINEL if body.app_error else body.payload.encode()
    cmd = CommandMessage(
        command=CommandType.DELIVER_ROUTE_PLAN,
        payload={
            "resolved_route": route.model_dump(mode="json"),
            "payload_b64": base64.b64encode(echo_payload).decode(),
            "connect_budget_sec": network.connect_budget_sec,
        },
    )
    try:
        resp = await node_registry.exec_node_cmd(
            body.origin_node_id, cmd, timeout=_ECHO_TIMEOUT_SEC
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        )
    if not resp.success or resp.data is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=resp.message or "route-plan delivery failed",
        )

    observations = [
        (Transport(item["transport"]), RouteObservationOutcome(item["outcome"]))
        for item in resp.data.get("observations", [])
    ]
    network.record_observations(origin, listener, observations)

    echoed_b64 = resp.data.get("echoed_b64")
    echoed = base64.b64decode(echoed_b64).decode() if echoed_b64 else None
    return NetworkEchoResponse(
        selected_transport=resp.data.get("selected_transport"),
        echoed=echoed,
        route_epoch=route.route_epoch,
        candidates=[candidate.transport.value for candidate in route.candidates],
        reachability=network.reachability_states(origin, listener),
    )


@router.get(
    "/endpoints",
    summary="List network endpoint advertisements",
    description="List the advertised node network-plane endpoints.",
)
async def list_network_endpoints(
    principal: PrincipalContext = Depends(authenticate_connection),
    plane: NetworkPlane | None = Depends(get_network_plane),
    logger: logging.Logger = Depends(get_logger),
) -> list[NetworkEndpointInfo]:
    await _require_admin(principal, logger)
    network = _require_plane(plane)
    return [
        NetworkEndpointInfo(
            endpoint_id=adv.endpoint_id,
            node_id=adv.node_id,
            url=adv.url,
            generation=adv.generation,
            trust_domain=adv.trust_domain,
            reachability_class=adv.reachability_class.value,
            protocols=list(adv.protocols),
        )
        for adv in await network.endpoints()
    ]


@router.get(
    "/reachability",
    summary="List derived reachability evidence",
    description="List the directional reachability entries derived from observations.",
)
async def list_network_reachability(
    principal: PrincipalContext = Depends(authenticate_connection),
    plane: NetworkPlane | None = Depends(get_network_plane),
    logger: logging.Logger = Depends(get_logger),
) -> list[NetworkReachabilityEntryInfo]:
    await _require_admin(principal, logger)
    network = _require_plane(plane)
    return [
        NetworkReachabilityEntryInfo(
            origin_id=str(entry["origin_id"]),
            target_node_id=str(entry["target_node_id"]),
            incarnation=int(entry["incarnation"]),
            listener_generation=int(entry["listener_generation"]),
            transport=str(entry["transport"]),
            state=str(entry["state"]),
            retries=int(entry["retries"]),
        )
        for entry in network.reachability_snapshot()
    ]

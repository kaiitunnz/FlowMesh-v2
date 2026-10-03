import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi import Path as ApiPath
from fastapi import WebSocket, WebSocketDisconnect, status

from shared.network.byte_stream import ByteStreamChannel, StreamClosed
from shared.utils.json import safe_get

from ...app_state import (
    get_logger,
    get_runtime,
    get_ssh_connection_registry,
    get_ssh_proxy_enabled,
    get_ssh_relay,
    get_worker_registry,
)
from ...auth.security import (
    PrincipalContext,
    authenticate_connection,
    authenticate_websocket,
    require_permission,
)
from ...hooks import ResourceAction, ResourceKind
from ...registries.worker import WorkerRegistry
from ...schemas.ssh import SSHConnectionInfo
from ...services.ssh_connections import SshConnectionRegistry
from ...ssh import SshRelayOrigin, resolve_relay_target
from ...ssh.connections import tracked_ssh_connection
from ...task.runtime import TaskRuntime
from ._listing import ListFilter, filter_params

router = APIRouter(prefix="/ssh", tags=["SSH"])


@router.websocket("/tasks/{task_id}/proxy")
async def ssh_proxy(
    websocket: WebSocket,
    task_id: str = ApiPath(..., min_length=1),
    principal: PrincipalContext = Depends(authenticate_websocket),
    runtime: TaskRuntime = Depends(get_runtime),
    logger: logging.Logger = Depends(get_logger),
    proxy_enabled: bool = Depends(get_ssh_proxy_enabled),
    worker_registry: WorkerRegistry = Depends(get_worker_registry),
    ssh_relay: SshRelayOrigin | None = Depends(get_ssh_relay),
    ssh_connections: SshConnectionRegistry | None = Depends(
        get_ssh_connection_registry
    ),
) -> None:
    """Proxy an SSH session over WebSocket.

    Connects a client to a running SSH task whose session is published in a relayed
    mode, carrying its bytes over the network plane's relay to the worker serving it.

    Authentication: bearer token from the ``Authorization`` header, or
    ``?token=...`` query param for browser clients that can't set headers.

    Close behavior:
    - ``4401``: missing or invalid bearer token
    - ``4403``: proxy access disabled, or principal not authorized for the task
    - ``4404``: task not found
    - ``1011``: the session cannot be relayed
    """
    try:
        await require_permission(
            principal, ResourceKind.TASK, task_id, ResourceAction.READ, logger
        )
    except Exception:
        await websocket.close(code=4403, reason="forbidden")
        return

    if not proxy_enabled or ssh_relay is None:
        await websocket.close(code=4403, reason="proxy disabled")
        return

    record = runtime.get_record(task_id)
    if record is None:
        await websocket.close(code=4404, reason="task not found")
        return

    try:
        target = await resolve_relay_target(record, worker_registry)
        channel = await ssh_relay.open(target)
    except Exception as exc:
        logger.warning("Cannot relay SSH for task %s: %s", task_id, exc)
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR, reason="relay unavailable"
        )
        return

    username = safe_get(record.latest_update, "ssh.username")
    client = websocket.client
    try:
        await websocket.accept()
        async with tracked_ssh_connection(
            ssh_connections,
            "proxy",
            target,
            record.workflow_id,
            str(username) if username else None,
            (None, None) if client is None else (client.host, client.port),
            logger,
        ):
            logger.info("SSH relay started: task=%s", task_id)
            await _relay_websocket(websocket, channel)
    finally:
        await ssh_relay.release(channel, abort=True)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("SSH relay ended: task=%s", task_id)


async def _relay_websocket(websocket: WebSocket, channel: ByteStreamChannel) -> None:
    """Carry bytes both ways until either end closes.

    A WebSocket has no half-close, so the first end to close ends the connection.
    """

    async def relay_to_client() -> None:
        while (data := await channel.recv()) is not None:
            await websocket.send_bytes(data)

    async def client_to_relay() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if (data := message.get("bytes")) is None:
                await websocket.close(
                    code=status.WS_1003_UNSUPPORTED_DATA, reason="binary frames only"
                )
                return
            await channel.send(data)

    tasks = {
        asyncio.ensure_future(relay_to_client()),
        asyncio.ensure_future(client_to_relay()),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except (StreamClosed, WebSocketDisconnect, RuntimeError):
                pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


SSH_CONNECTION_FILTER_FIELDS = frozenset(
    {
        "connection_id",
        "session_id",
        "access_mode",
        "task_id",
        "workflow_id",
        "worker_id",
        "node_id",
        "username",
        "source_ip",
        "source_port",
    }
)


@router.get(
    "/connections",
    summary="List SSH connections",
    description="List active SSH connections.",
    response_description="List of active SSH connection records.",
)
async def list_ssh_connections(
    principal: PrincipalContext = Depends(authenticate_connection),
    filters: ListFilter = Depends(filter_params(SSH_CONNECTION_FILTER_FIELDS)),
    ssh_connections: SshConnectionRegistry | None = Depends(
        get_ssh_connection_registry
    ),
    logger: logging.Logger = Depends(get_logger),
) -> list[SSHConnectionInfo]:
    await require_permission(
        principal, ResourceKind.SYSTEM, None, ResourceAction.ADMIN, logger
    )
    query = filters.parse()
    if ssh_connections is None:
        return []
    connections = await ssh_connections.list_connections()
    return query.filter(connections)

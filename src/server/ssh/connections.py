"""Records each relayed SSH client connection while it lasts."""

import contextlib
import logging
from collections.abc import AsyncIterator

from shared.utils import new_ssh_connection_id, now_iso

from ..schemas.ssh import SSHConnectionInfo
from ..services.ssh_connections import SshConnectionRegistry
from .relay import SshRelayTarget


@contextlib.asynccontextmanager
async def tracked_ssh_connection(
    registry: SshConnectionRegistry | None,
    access_mode: str,
    target: SshRelayTarget,
    workflow_id: str | None,
    username: str | None,
    peer: tuple[str | None, int | None],
    logger: logging.Logger,
) -> AsyncIterator[None]:
    """List one client connection in ``registry`` for as long as the block runs."""
    if registry is None:
        yield
        return
    connection_id = new_ssh_connection_id()
    source_ip, source_port = peer
    try:
        await registry.register_connection(
            SSHConnectionInfo(
                connection_id=connection_id,
                access_mode=access_mode,
                task_id=target.task_id,
                workflow_id=workflow_id,
                worker_id=target.worker_id,
                node_id=target.node_id,
                session_id=target.endpoint_id,
                username=username,
                source_ip=source_ip,
                source_port=source_port,
                connected_at=now_iso(),
            )
        )
    except Exception:
        logger.debug(
            "Failed to register SSH connection %s", connection_id, exc_info=True
        )
    try:
        yield
    finally:
        try:
            await registry.unregister_connection(connection_id)
        except Exception:
            logger.debug(
                "Failed to unregister SSH connection %s", connection_id, exc_info=True
            )

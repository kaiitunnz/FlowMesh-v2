"""Stops the root's SSH ingresses for a root shutdown."""

from ..services.port_forward import PortForwardService
from .relay import SshRelayOrigin


async def stop_ssh_ingresses(
    origin: SshRelayOrigin | None, forward: PortForwardService | None
) -> None:
    """Stop the SSH relay origin, then the forward listener, before the root's
    bridge pumps stop.

    The origin ends each live connection at its worker directly, so no forward
    connection that ends afterwards leaves its cancel to a pump about to stop or
    shortens the record the next root reaps.
    """
    if origin is not None:
        await origin.stop()
    if forward is not None:
        await forward.stop()

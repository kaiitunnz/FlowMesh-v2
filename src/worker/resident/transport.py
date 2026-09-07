"""The frame sink a worker's resident session emits through.

A session hands each frame it produces to this sink; the worker runtime carries the
frame to its own supervisor over the authenticated attachment, which relays it opaquely
over the reverse-rendezvous relay to the peer worker. Inbound frames are pushed into the
session by the runtime through ``ResidentRelaySession.on_frame``.
"""

from collections.abc import Awaitable, Callable
from typing import Protocol

from shared.network.relay_frame import RelayFrame


class ResidentFrameSink(Protocol):
    """Carries one produced relay frame toward the peer worker."""

    async def send(self, frame: RelayFrame) -> None: ...


FrameSend = Callable[[RelayFrame], Awaitable[None]]

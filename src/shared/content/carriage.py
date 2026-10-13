"""The routes a content transfer is carried over, resolved by control beside its grant.

A carriage plan is routing evidence, never authority: the grant alone decides what may
be read, and a plan names only how the requester reaches the holder control chose for
it — the peer transports it may dial, in the order to try them, and the root relay
otherwise. It travels beside the grant, so refreshing a route leaves the grant the
holder compares untouched. A requester honors it for ``fresh_for_sec`` on its own clock
from arrival, never past the grant's expiry.
"""

import time

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas.network import Transport


class ContentPeerRoute(BaseModel):
    """One peer transport the requester may dial, and the address it dials."""

    model_config = ConfigDict(frozen=True)

    transport: Transport
    endpoint: str


class ContentCarriagePlan(BaseModel):
    """How the requester reaches the holder for one granted transfer.

    The grant, holder, listener generation and route epoch are carried for diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    transfer_session_id: str
    grant_id: str
    holder_worker_id: str
    holder_generation: int
    listener_generation: int
    # Peer transports to dial, best first; with none left, the transfer takes the root
    # relay where the read allows it.
    routes: tuple[ContentPeerRoute, ...] = ()
    route_epoch: int = 0
    fresh_for_sec: float = Field(default=0.0, ge=0.0)
    # Stamped by the requester on arrival, on its own monotonic clock.
    received_at: float = Field(default_factory=time.monotonic, exclude=True)

    @property
    def dialed(self) -> bool:
        """Whether the requester dials the holder's side itself."""
        return bool(self.routes)

    def fresh(self, now: float | None = None) -> bool:
        """Whether the plan may still be dialed, on the requester's own clock."""
        elapsed = (time.monotonic() if now is None else now) - self.received_at
        return elapsed <= self.fresh_for_sec


class ContentRouteObservation(BaseModel):
    """One dialed transfer's path outcome, reported to control by session; control maps
    the session to its origin and holder."""

    model_config = ConfigDict(frozen=True)

    transfer_session_id: str
    transport: str
    outcome: str


__all__ = [
    "ContentCarriagePlan",
    "ContentPeerRoute",
    "ContentRouteObservation",
]

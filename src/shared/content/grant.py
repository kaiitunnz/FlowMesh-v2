"""The control-minted authorization for one bounded content hydration.

A worker that needs an object it does not hold asks the control plane, which checks the
requester's authenticated subject and the consumer binding that entitles it to this
reference, then mints a grant for exactly that object and relays it to both ends over
their authenticated attachments: the holder learns it may serve this object once, to
this requester, on this transfer, and the requester gets the token to present. A holder
serves nothing it was not granted, so naming an object is never enough to read it.

The grant is a physical read authorization and nothing else. It mints no invocation, no
service claim, no capacity credit, no route authorization, and no idempotency key; the
route a transfer takes is evidence about where the bytes are, never why they may be
read.
"""

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .reference import ContentReference

GRANT_VERSION = 1


class ContentOperation(StrEnum):
    """What a grant authorizes. Reading an immutable object is the only one."""

    HYDRATE = "hydrate"


class GrantRejection(StrEnum):
    """Why a holder refused a grant, all authorization rather than path failures."""

    UNKNOWN_GRANT = "unknown_grant"
    ALTERED_GRANT = "altered_grant"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    WRONG_HOLDER = "wrong_holder"
    STALE_GENERATION = "stale_generation"
    WRONG_OPERATION = "wrong_operation"
    UNSUPPORTED_VERSION = "unsupported_version"


class ContentHydrationGrant(BaseModel):
    """One holder's authorization to serve one object to one requester, once.

    ``reference`` is the exact object: a grant for one digest never opens another, and
    scope travels with it. ``holder_id`` and ``holder_generation`` bind the audience to
    the incarnation control resolved, so a grant minted for a worker that has since
    restarted is refused rather than served by its successor. ``transfer_session_id``
    is the relay session the bytes flow over, and ``expires_at_epoch`` bounds the whole
    exchange; a retry past it takes a fresh grant for the same immutable reference.
    """

    model_config = ConfigDict(frozen=True)

    grant_id: str
    version: int = GRANT_VERSION
    operation: ContentOperation = ContentOperation.HYDRATE
    reference: ContentReference
    requester_subject: str
    requester_origin_id: str
    holder_id: str
    holder_generation: int
    transfer_session_id: str
    expires_at_epoch: float

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) > self.expires_at_epoch


class HolderGrantGate:
    """A holder's one-use register of the grants control pre-delivered to it.

    Admission is the register, not the token: a grant the holder was never handed is
    refused whatever it claims, a grant that does not match the one it was handed is
    refused as altered, and a grant it has already served is refused as consumed. The
    gate authorizes; reading the object and verifying it against the reference is the
    caller's next step.
    """

    def __init__(self, *, holder_id: str, generation: int) -> None:
        self._holder_id = holder_id
        self._generation = generation
        self._granted: dict[str, ContentHydrationGrant] = {}
        self._consumed: dict[str, float] = {}

    def accept(self, grant: ContentHydrationGrant) -> None:
        """Register a grant control minted for this holder."""
        self._granted[grant.grant_id] = grant

    def admit(self, presented: ContentHydrationGrant) -> GrantRejection | None:
        """Consume the presented grant, or say why it is refused.

        The lookup runs before the sweep so an expired grant is refused as expired
        rather than as one this holder never had.
        """
        try:
            return self._decide(presented)
        finally:
            self._expire()

    def _decide(self, presented: ContentHydrationGrant) -> GrantRejection | None:
        if presented.version != GRANT_VERSION:
            return GrantRejection.UNSUPPORTED_VERSION
        if presented.operation is not ContentOperation.HYDRATE:
            return GrantRejection.WRONG_OPERATION
        if presented.grant_id in self._consumed:
            return GrantRejection.CONSUMED
        held = self._granted.get(presented.grant_id)
        if held is None:
            return GrantRejection.UNKNOWN_GRANT
        if held != presented:
            return GrantRejection.ALTERED_GRANT
        if held.holder_id != self._holder_id:
            return GrantRejection.WRONG_HOLDER
        if held.holder_generation != self._generation:
            return GrantRejection.STALE_GENERATION
        if held.expired():
            return GrantRejection.EXPIRED
        del self._granted[presented.grant_id]
        self._consumed[presented.grant_id] = held.expires_at_epoch
        return None

    def _expire(self) -> None:
        now = time.time()
        for grant_id, grant in list(self._granted.items()):
            if grant.expired(now):
                del self._granted[grant_id]
        for grant_id, expiry in list(self._consumed.items()):
            if now > expiry:
                del self._consumed[grant_id]


__all__ = [
    "GRANT_VERSION",
    "ContentHydrationGrant",
    "ContentOperation",
    "GrantRejection",
    "HolderGrantGate",
]

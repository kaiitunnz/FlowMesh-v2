"""Durable ingress control facts.

An ingress invocation has no ``DS`` activation, continuation, or result slot. Its
terminal is a durable ingress-terminal fact keyed by ``invocation_id``, which the
Admission controller consumes to release the resident credit exactly as it consumes a
workflow's ``DS`` outcome. These facts are ``CS`` state, persisted and rehydrated so a
restart reconciles an accepted ingress claim against its recorded terminal.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..utils.time import now_iso


class IngressTerminalStatus(StrEnum):
    """The settled outcome of an ingress invocation."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngressTerminal(BaseModel):
    """The durable terminal fact for one ingress invocation.

    Recorded only from the designated worker's fenced outcome (a completion, a definite
    failure, or a reap), it is the fenced record the Admission controller consumes by
    ``invocation_id``. A client disconnect, stream close, or telemetry report alone
    never records it, so none of those releases the credit.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    status: IngressTerminalStatus
    detail: str | None = None
    created_at: str = Field(default_factory=now_iso)


class IngressSnapshot(BaseModel):
    """The persisted ingress control facts."""

    terminals: list[IngressTerminal] = Field(default_factory=list)


class IngressTerminalStore:
    """Durable, idempotent custody of ingress-terminal facts by ``invocation_id``."""

    def __init__(self) -> None:
        self._terminals: dict[str, IngressTerminal] = {}

    def record(self, terminal: IngressTerminal) -> bool:
        """Record a terminal for an invocation; return False if one already exists.

        The first fenced terminal wins and is authoritative; a duplicate or late report
        is a no-op so a settled credit is never released twice.
        """
        if terminal.invocation_id in self._terminals:
            return False
        self._terminals[terminal.invocation_id] = terminal
        return True

    def get(self, invocation_id: str) -> IngressTerminal | None:
        return self._terminals.get(invocation_id)

    def all(self) -> list[IngressTerminal]:
        return list(self._terminals.values())

    def to_snapshot(self) -> IngressSnapshot:
        return IngressSnapshot(terminals=list(self._terminals.values()))

    def load_snapshot(self, snapshot: IngressSnapshot) -> None:
        self._terminals = {t.invocation_id: t for t in snapshot.terminals}

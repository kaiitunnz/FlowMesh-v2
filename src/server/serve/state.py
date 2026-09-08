"""Durable external status-terminal facts for task-addressed serve invocations.

A task-addressed external invocation has no ``DS`` activation, continuation, or result
slot. Its terminal is a durable status fact keyed by ``invocation_id``, which the
Admission controller consumes to release the resident credit exactly as it consumes a
workflow's ``DS`` outcome. These facts are ``CS`` state, persisted and rehydrated so a
restart reconciles an accepted external claim against its recorded terminal.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..utils.time import now_iso


class ServeTerminalStatus(StrEnum):
    """The settled outcome of a task-addressed external serve invocation."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServeStatusTerminal(BaseModel):
    """The durable external status-terminal fact for one serve invocation.

    Recorded only from the replica sidecar's fence-matching status terminal (a
    completion, a definite failure, or a reap), it is the fenced record the Admission
    controller consumes by ``invocation_id``. A client disconnect, stream close,
    relay-window acknowledgement, or timer alone never records it, so none of those
    releases the credit.
    """

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    status: ServeTerminalStatus
    detail: str | None = None
    created_at: str = Field(default_factory=now_iso)


class ServeTerminalSnapshot(BaseModel):
    """The persisted external status-terminal facts."""

    terminals: list[ServeStatusTerminal] = Field(default_factory=list)


class ServeTerminalStore:
    """Durable, idempotent custody of external status-terminal facts by
    ``invocation_id``."""

    def __init__(self) -> None:
        self._terminals: dict[str, ServeStatusTerminal] = {}

    def record(self, terminal: ServeStatusTerminal) -> bool:
        """Record a terminal for an invocation; return False if one already exists.

        The first fence-matching terminal wins and is authoritative; a duplicate or late
        report is a no-op so a settled credit is never released twice.
        """
        if terminal.invocation_id in self._terminals:
            return False
        self._terminals[terminal.invocation_id] = terminal
        return True

    def get(self, invocation_id: str) -> ServeStatusTerminal | None:
        return self._terminals.get(invocation_id)

    def all(self) -> list[ServeStatusTerminal]:
        return list(self._terminals.values())

    def to_snapshot(self) -> ServeTerminalSnapshot:
        return ServeTerminalSnapshot(terminals=list(self._terminals.values()))

    def load_snapshot(self, snapshot: ServeTerminalSnapshot) -> None:
        self._terminals = {t.invocation_id: t for t in snapshot.terminals}

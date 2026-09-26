"""A workflow's published outputs, as clients name and page them.

A published output is named by the node its author declared it on. A per-node output
is one member; a published spawn is a family of keyed collections, one per scope that
spawns, whose members are keyed by child index within their scope. Members order by
that stable identity — never by when they settled — so a cursor names a position that
does not move.
"""

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

from ..orchestration import OrchestrationEngine
from ..orchestration.state import ResultPublication
from .v2.representations.results import CardinalityKind, ResultDeclaration


class InvalidCursor(ValueError):
    """A cursor that is not one this catalog issued."""


@dataclass(frozen=True)
class OutputMember:
    """One member of a published output: a singleton, or one collection member."""

    name: str
    declaration: ResultDeclaration
    scope_id: str | None
    key: str | None
    sequence: int | None
    publication: ResultPublication | None

    @property
    def keyed(self) -> bool:
        return self.declaration.cardinality is CardinalityKind.KEYED_COLLECTION

    @property
    def cursor(self) -> str:
        identity = [self.name, self.scope_id, self.key, self.sequence]
        raw = json.dumps(identity, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode("ascii")

    @property
    def order(self) -> tuple[Any, ...]:
        return _order(self.name, self.scope_id, self.key, self.sequence)


@dataclass(frozen=True)
class PublishedOutputs:
    """A workflow's published outputs as of one look at its ledger."""

    org_id: str
    members: list[OutputMember]
    # Whether the workflow can publish more; a last page is final only when it cannot.
    open: bool


def published_members(engine: OrchestrationEngine) -> list[OutputMember]:
    """Every member a workflow's published outputs hold so far, in cursor order."""
    members = [
        OutputMember(
            name=name,
            declaration=decl,
            scope_id=slot.scope_id,
            key=slot.logical_key,
            sequence=slot.sequence,
            publication=engine.output_publication(
                decl.output_id, slot.scope_id, slot.logical_key, slot.sequence
            ),
        )
        for name, decl in engine.published_outputs()
        for slot in engine.output_slots(decl.output_id)
    ]
    return sorted(members, key=lambda member: member.order)


def page(
    members: list[OutputMember],
    limit: int,
    after: str | None = None,
    before: str | None = None,
) -> list[OutputMember]:
    """The ``limit`` members strictly after, or strictly before, a cursor."""
    if after is not None:
        bound = decode_cursor(after)
        return [m for m in members if m.order > bound][:limit]
    if before is not None:
        bound = decode_cursor(before)
        return [m for m in members if m.order < bound][-limit:]
    return members[:limit]


def decode_cursor(cursor: str) -> tuple[Any, ...]:
    try:
        identity = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        name, scope_id, key, sequence = identity
    except (binascii.Error, UnicodeError, ValueError, TypeError) as exc:
        raise InvalidCursor(f"invalid cursor {cursor!r}") from exc
    if not isinstance(name, str) or not all(
        value is None or isinstance(value, str) for value in (scope_id, key)
    ):
        raise InvalidCursor(f"invalid cursor {cursor!r}")
    if sequence is not None and not isinstance(sequence, int):
        raise InvalidCursor(f"invalid cursor {cursor!r}")
    return _order(name, scope_id, key, sequence)


def _order(
    name: str, scope_id: str | None, key: str | None, sequence: int | None
) -> tuple[Any, ...]:
    # A child index orders numerically, so member 10 follows member 9.
    numeric = key is not None and key.isascii() and key.isdigit()
    key_order = (0, int(key), "") if numeric and key else (1, 0, key or "")
    return (name, scope_id or "", key_order, -1 if sequence is None else sequence)


__all__ = [
    "InvalidCursor",
    "OutputMember",
    "PublishedOutputs",
    "decode_cursor",
    "page",
    "published_members",
]

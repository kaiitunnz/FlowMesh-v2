"""A workflow's published outputs, as clients name and page them.

A published output is named by the node its author declared it on. A per-node output
is one member; a published spawn is a family of keyed collections, one per scope that
spawns, whose members are keyed by child index within their scope. Members order by
that stable identity, so a cursor names a fixed position.
"""

from dataclasses import dataclass
from typing import Any

from ..orchestration import OrchestrationEngine
from ..orchestration.state import ResultPublication, ResultSlot
from ..utils.cursors import InvalidCursor, decode_cursor, encode_cursor, page_slice
from .v2.representations.results import CardinalityKind, ResultDeclaration


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
        return encode_cursor([self.name, self.scope_id, self.key, self.sequence])

    @property
    def order(self) -> tuple[Any, ...]:
        return _order(self.name, self.scope_id, self.key, self.sequence)


@dataclass(frozen=True)
class PublishedOutputs:
    """A workflow's published outputs as of one look at its ledger."""

    members: list[OutputMember]
    # Whether the workflow can publish more; a last page is final only when it cannot.
    open: bool


@dataclass(frozen=True)
class PublishedOutput:
    """One named published output and, when it holds one, the member selected."""

    declaration: ResultDeclaration | None
    member: OutputMember | None
    open: bool


def published_members(
    engine: OrchestrationEngine, name: str | None = None
) -> list[OutputMember]:
    """Every member a workflow's published outputs hold so far, or one output's."""
    return [
        _member(engine, published, decl, slot)
        for published, decl in engine.published_outputs()
        if name is None or published == name
        for slot in engine.output_slots(decl.output_id)
    ]


def published_member(
    engine: OrchestrationEngine,
    name: str,
    scope_id: str | None,
    key: str | None,
    sequence: int | None,
) -> tuple[ResultDeclaration | None, OutputMember | None]:
    """The declaration a public name refers to, and its member at the selectors."""
    decl = next((d for n, d in engine.published_outputs() if n == name), None)
    if decl is None:
        return None, None
    slot = engine.output_slot(decl.output_id, scope_id, key, sequence)
    return decl, _member(engine, name, decl, slot) if slot is not None else None


def paginate_members(
    members: list[OutputMember],
    limit: int,
    after: tuple[Any, ...] | None = None,
    before: tuple[Any, ...] | None = None,
) -> list[OutputMember]:
    """Select the ``limit`` members strictly after, or strictly before, a decoded
    cursor position."""
    ordered = sorted(members, key=lambda member: member.order)
    window = page_slice(
        [member.order for member in ordered], limit, after=after, before=before
    )
    return ordered[window]


def _member(
    engine: OrchestrationEngine, name: str, decl: ResultDeclaration, slot: ResultSlot
) -> OutputMember:
    return OutputMember(
        name=name,
        declaration=decl,
        scope_id=slot.scope_id,
        key=slot.logical_key,
        sequence=slot.sequence,
        publication=engine.output_publication(
            decl.output_id, slot.scope_id, slot.logical_key, slot.sequence
        ),
    )


def decode_member_cursor(cursor: str) -> tuple[Any, ...]:
    """Return the member position a cursor encodes; raise InvalidCursor otherwise."""
    match decode_cursor(cursor):
        case [
            str() as name,
            str() | None as scope_id,
            str() | None as key,
            int() | None as sequence,
        ] if not isinstance(sequence, bool):
            return _order(name, scope_id, key, sequence)
    raise InvalidCursor(f"invalid cursor {cursor!r}")


def _order(
    name: str, scope_id: str | None, key: str | None, sequence: int | None
) -> tuple[Any, ...]:
    # A child index orders numerically, so member 10 follows member 9: by its digit
    # count without leading zeros, then by its digits.
    if key and key.isascii() and key.isdigit():
        digits = key.lstrip("0")
        key_order: tuple[int, int, str] = (0, len(digits), digits)
    else:
        key_order = (1, 0, key or "")
    return (name, scope_id or "", key_order, -1 if sequence is None else sequence)


__all__ = [
    "OutputMember",
    "PublishedOutput",
    "PublishedOutputs",
    "decode_member_cursor",
    "paginate_members",
    "published_member",
    "published_members",
]

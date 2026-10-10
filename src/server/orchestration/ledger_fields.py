"""The stored layout of an orchestration ledger: one field per ledger entity.

A ledger is stored as one hash per workflow. Each keyed entity is a field named by its
collection and its key, holding its insertion ordinal and its JSON; each history entry
is a field named by its position; each set member is a field of its own; and ``meta``
fields hold the instance's foundation, written once, and its scalars. A write carries
only the fields that changed, and a restore rebuilds the ledger snapshot from all of
them, refusing a field it cannot place.
"""

import json
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .journal import JournalKey
from .state import (
    AcceptedInput,
    Activation,
    Attempt,
    AuthorityDecision,
    BoundaryEvent,
    BranchDecision,
    ChildContext,
    Continuation,
    ControlState,
    DelegatedAuthorityGrant,
    EffectReceipt,
    EmbodimentSelection,
    InputPreparation,
    InputResolution,
    Invocation,
    IterationResolution,
    LedgerSnapshot,
    LoopInstance,
    Occurrence,
    OrchestrationEvent,
    PrivateStateLineage,
    ProgressCapability,
    Record,
    RegionJoinAggregate,
    ResultPublication,
    ResultSlot,
    Scope,
    WorkItem,
)

LAYOUT = "1"

# Each keyed collection's entity model and the key it is held under.
KEYED: Mapping[str, tuple[type[BaseModel], Callable[[Any], Hashable]]] = {
    "scopes": (Scope, lambda e: e.scope_id),
    "activations": (Activation, lambda e: e.activation_id),
    "work_items": (WorkItem, lambda e: e.work_item_id),
    "continuations": (Continuation, lambda e: e.work_item_id),
    "invocations": (Invocation, lambda e: e.invocation_id),
    "attempts": (Attempt, lambda e: e.attempt_id),
    "embodiment_selections": (EmbodimentSelection, lambda e: e.work_item_id),
    "input_resolutions": (InputResolution, lambda e: e.work_item_id),
    "input_preparations": (InputPreparation, lambda e: e.work_item_id),
    "boundary_events": (BoundaryEvent, lambda e: (e.activation, e.call_correlation)),
    "effect_receipts": (EffectReceipt, lambda e: e.invocation_id),
    "delegated_grants": (DelegatedAuthorityGrant, lambda e: e.grant_id),
    "progress_capabilities": (ProgressCapability, lambda e: (e.scope_id, e.axis)),
    "result_slots": (ResultSlot, lambda e: e.slot_key),
    "result_publications": (ResultPublication, lambda e: e.slot_key),
    "private_state": (PrivateStateLineage, lambda e: e.binding.reference.activation_id),
    "occurrences": (Occurrence, lambda e: e.key),
    "control_states": (ControlState, lambda e: e.key),
    "branch_decisions": (BranchDecision, lambda e: e.occurrence),
    "loop_instances": (LoopInstance, lambda e: e.scope_id),
    "iteration_resolutions": (IterationResolution, lambda e: (e.loop, e.iteration)),
    "child_contexts": (ChildContext, lambda e: e.context_id),
}
# Keyed collections of plain strings, keyed by task id.
STRINGS = ("failure_reasons",)
HISTORIES: Mapping[str, type[BaseModel]] = {
    "records": Record,
    "accepted_inputs": AcceptedInput,
    "region_aggregates": RegionJoinAggregate,
    "authority_decisions": AuthorityDecision,
    "trace": OrchestrationEvent,
}
SETS = ("released_scopes", "failed_regions", "failed_scopes")
# Written with the whole ledger only: they never change once the instance is built.
FOUNDATION = ("instance", "root_scope", "root_grant", "max_loop_iterations")
SCALARS = ("next_seq", "instance_failure", "control_failure", "instance_cancelled")

META = "meta"
_LAYOUT_FIELD = f"{META}:layout"
_FOUNDATION_FIELD = f"{META}:foundation"
SET_MEMBER = "1"


type LedgerOrdinals = Mapping[str, Mapping[Hashable, int]]
type KeyedEntries = Iterable[tuple[Hashable, int, BaseModel | str]]


@dataclass(frozen=True)
class LedgerChanges:
    """The fields one ledger write sets and deletes, and whether it first drops the
    whole stored ledger, with what of the ledger it captured: the journal's changes at
    their versions, each history's length, the scalars, and the rewrite it makes."""

    fields: dict[str, str]
    deleted: tuple[str, ...] = ()
    reset: bool = False
    captured: Mapping[JournalKey, int] = field(default_factory=dict)
    lengths: Mapping[str, int] = field(default_factory=dict)
    scalars: Mapping[str, Any] = field(default_factory=dict)
    rewrite: int = 0


@dataclass(frozen=True)
class StoredLedger:
    """A stored ledger as a snapshot, with each keyed entity's insertion ordinal."""

    snapshot: LedgerSnapshot
    ordinals: LedgerOrdinals


def _key_part(part: Hashable) -> Any:
    return part.value if isinstance(part, Enum) else part


def field_name(collection: str, key: Hashable) -> str:
    """The field of one entry: its collection, then its key's components as a JSON
    array, so distinct keys of distinct types never share a field."""
    parts = key if isinstance(key, tuple) else (key,)
    encoded = json.dumps([_key_part(p) for p in parts], separators=(",", ":"))
    return f"{collection}:{encoded}"


def history_field(collection: str, position: int) -> str:
    return field_name(collection, position)


def member_field(collection: str, member: Hashable) -> str:
    return field_name(collection, member)


def keyed_value(ordinal: int, value: BaseModel | str) -> str:
    encoded = (
        value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value)
    )
    return f"{ordinal}:{encoded}"


def scalar_field(name: str) -> str:
    return f"{META}:{name}"


def foundation_fields(snapshot_fields: Mapping[str, Any]) -> dict[str, str]:
    """The fields of a ledger's layout and foundation."""
    foundation = {
        name: (value.model_dump(mode="json") if isinstance(value, BaseModel) else value)
        for name, value in snapshot_fields.items()
    }
    return {
        _LAYOUT_FIELD: LAYOUT,
        _FOUNDATION_FIELD: json.dumps(foundation, separators=(",", ":")),
    }


def scalar_value(value: Any) -> str:
    return json.dumps(value)


def encode_fields(
    foundation: Mapping[str, Any],
    keyed: Mapping[str, KeyedEntries],
    histories: Mapping[str, Sequence[BaseModel]],
    sets: Mapping[str, Iterable[str]],
    scalars: Mapping[str, Any],
) -> dict[str, str]:
    """Every field of a ledger."""
    fields = foundation_fields(foundation)
    for collection, entries in keyed.items():
        for key, ordinal, value in entries:
            fields[field_name(collection, key)] = keyed_value(ordinal, value)
    for collection, history in histories.items():
        for position, entry in enumerate(history):
            fields[history_field(collection, position)] = entry.model_dump_json()
    for collection, members in sets.items():
        for member in members:
            fields[member_field(collection, member)] = SET_MEMBER
    for name, value in scalars.items():
        fields[scalar_field(name)] = scalar_value(value)
    return fields


def encode_ledger(stored: StoredLedger) -> dict[str, str]:
    """Every field of a stored ledger."""
    snapshot, ordinals = stored.snapshot, stored.ordinals

    def keyed(collection: str) -> list[tuple[Hashable, int, BaseModel | str]]:
        order = ordinals.get(collection, {})
        if collection in STRINGS:
            reasons: dict[str, str] = getattr(snapshot, collection)
            return [(key, order[key], value) for key, value in reasons.items()]
        key_of = KEYED[collection][1]
        entities: list[BaseModel] = getattr(snapshot, collection)
        return [(key_of(e), order[key_of(e)], e) for e in entities]

    return encode_fields(
        {name: getattr(snapshot, name) for name in FOUNDATION},
        {collection: keyed(collection) for collection in (*KEYED, *STRINGS)},
        {collection: getattr(snapshot, collection) for collection in HISTORIES},
        {collection: getattr(snapshot, collection) for collection in SETS},
        {name: getattr(snapshot, name) for name in SCALARS},
    )


class LedgerLayoutError(ValueError):
    """A stored ledger holds a field this layout cannot place."""


def _parse_key(collection: str, encoded: str) -> list[Any]:
    try:
        parts = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise LedgerLayoutError(f"malformed key in {collection}: {encoded!r}") from exc
    if not isinstance(parts, list) or not parts:
        raise LedgerLayoutError(f"malformed key in {collection}: {encoded!r}")
    return parts


def _split_ordinal(collection: str, value: str) -> tuple[int, str]:
    ordinal, sep, encoded = value.partition(":")
    if not sep or not ordinal.isdigit():
        raise LedgerLayoutError(f"entry of {collection} has no ordinal")
    return int(ordinal), encoded


def _names_member(collection: str, parts: list[Any], name: str) -> bool:
    return (
        len(parts) == 1
        and isinstance(parts[0], str)
        and field_name(collection, parts[0]) == name
    )


def decode_ledger(fields: Mapping[str, str]) -> StoredLedger:
    """Rebuild a stored ledger from its fields, refusing one that does not hold
    exactly the fields its layout places."""
    try:
        return _decode_ledger(fields)
    except LedgerLayoutError:
        raise
    except ValueError as exc:
        raise LedgerLayoutError(f"unreadable ledger: {exc}") from exc


def _decode_ledger(fields: Mapping[str, str]) -> StoredLedger:
    if fields.get(_LAYOUT_FIELD) != LAYOUT:
        raise LedgerLayoutError(f"unknown ledger layout {fields.get(_LAYOUT_FIELD)!r}")
    if (foundation := fields.get(_FOUNDATION_FIELD)) is None:
        raise LedgerLayoutError("ledger has no foundation")
    data = json.loads(foundation)
    if not isinstance(data, dict) or data.keys() != set(FOUNDATION):
        raise LedgerLayoutError("ledger foundation does not hold its fields")
    keyed: dict[str, list[tuple[int, Any, Any]]] = {}
    histories: dict[str, dict[int, Any]] = {}
    sets: dict[str, list[Any]] = {}
    scalars: dict[str, Any] = {}
    for name, value in fields.items():
        collection, sep, encoded = name.partition(":")
        if not sep:
            raise LedgerLayoutError(f"malformed field {name!r}")
        if collection == META:
            if encoded in SCALARS:
                scalars[encoded] = json.loads(value)
            elif name not in (_LAYOUT_FIELD, _FOUNDATION_FIELD):
                raise LedgerLayoutError(f"unknown ledger field {name!r}")
            continue
        parts = _parse_key(collection, encoded)
        if collection in KEYED:
            ordinal, body = _split_ordinal(collection, value)
            model, key_of = KEYED[collection]
            entity = model.model_validate_json(body)
            key = key_of(entity)
            if field_name(collection, key) != name:
                raise LedgerLayoutError(f"{name!r} holds the entry of {key!r}")
            keyed.setdefault(collection, []).append((ordinal, key, entity))
        elif collection in STRINGS:
            ordinal, body = _split_ordinal(collection, value)
            if not _names_member(collection, parts, name):
                raise LedgerLayoutError(f"malformed key {name!r}")
            keyed.setdefault(collection, []).append(
                (ordinal, parts[0], json.loads(body))
            )
        elif collection in HISTORIES:
            if (
                len(parts) != 1
                or type(parts[0]) is not int
                or history_field(collection, parts[0]) != name
            ):
                raise LedgerLayoutError(f"malformed position {name!r}")
            histories.setdefault(collection, {})[parts[0]] = HISTORIES[
                collection
            ].model_validate_json(value)
        elif collection in SETS:
            if not _names_member(collection, parts, name) or value != SET_MEMBER:
                raise LedgerLayoutError(f"malformed member {name!r}")
            sets.setdefault(collection, []).append(parts[0])
        else:
            raise LedgerLayoutError(f"unknown ledger collection {collection!r}")

    ordinals: dict[str, dict[Hashable, int]] = {}
    for collection, entries in keyed.items():
        entries.sort(key=lambda entry: entry[0])
        if len({ordinal for ordinal, _, _ in entries}) != len(entries):
            raise LedgerLayoutError(f"conflicting ordinals in {collection}")
        ordinals[collection] = {key: ordinal for ordinal, key, _ in entries}
        if collection in STRINGS:
            data[collection] = {key: value for _, key, value in entries}
        else:
            data[collection] = [entity for _, _, entity in entries]
    for collection, positions in histories.items():
        if sorted(positions) != list(range(len(positions))):
            raise LedgerLayoutError(f"{collection} is missing an entry")
        data[collection] = [positions[i] for i in range(len(positions))]
    for collection, members in sets.items():
        data[collection] = sorted(members)
    if scalars.keys() != set(SCALARS):
        raise LedgerLayoutError("ledger does not hold every scalar")
    data.update(scalars)
    return StoredLedger(LedgerSnapshot.model_validate(data), ordinals)

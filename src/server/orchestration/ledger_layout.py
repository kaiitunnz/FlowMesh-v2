"""The stored layout of an orchestration ledger: one field per ledger entity.

A ledger is stored as one hash per workflow. Each keyed entity is a field named by its
collection and its key, holding its insertion ordinal and its JSON; each history entry
is a field named by its position; each set member is a field of its own; and ``meta``
fields hold the instance's foundation, written only with the whole ledger, and its
scalars. A write carries only the fields that changed, and a restore rebuilds the
ledger snapshot from all of them, refusing a field it cannot place.
"""

import json
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from types import GenericAlias
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

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
KEYED_TEXTS = ("failure_reasons",)
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
# A keyed collection's entries by key, with each key's insertion ordinal.
type KeyedEntries = tuple[Mapping[Any, BaseModel | str], Mapping[Any, int]]


@dataclass(frozen=True)
class LedgerChanges:
    """One ledger write, and what of the ledger it captured."""

    fields: dict[str, str]
    """The fields the write sets."""
    deleted: tuple[str, ...] = ()
    """The fields the write deletes."""
    reset: bool = False
    """Whether the write first drops the whole stored ledger."""
    captured: Mapping[JournalKey, int] = field(default_factory=dict)
    """The journal's changes the write carries, at their versions."""
    lengths: Mapping[str, int] = field(default_factory=dict)
    """Each history's length when captured."""
    scalars: Mapping[str, Any] = field(default_factory=dict)
    """The scalars when captured."""
    rewrite: int = 0
    """The count of whole-ledger rewrites owed up to the write, which its landing
    settles; 0 for a write of changes only."""


@dataclass(frozen=True)
class StoredLedger:
    """A stored ledger as a snapshot, with each keyed entity's insertion ordinal."""

    snapshot: LedgerSnapshot
    ordinals: LedgerOrdinals


def applied_changes(
    fields: Mapping[str, str], changes: LedgerChanges
) -> dict[str, str]:
    """Return the fields a ledger stores once ``changes`` land on ``fields``."""
    applied = {} if changes.reset else dict(fields)
    applied.update(changes.fields)
    for name in changes.deleted:
        applied.pop(name, None)
    return applied


def _key_part(part: Hashable) -> Any:
    return part.value if isinstance(part, Enum) else part


def _encoded_part(part: Hashable) -> str:
    if type(part) is int:
        return str(part)
    if (
        type(part) is str
        and part.isascii()
        and part.isprintable()
        and '"' not in part
        and "\\" not in part
    ):
        return f'"{part}"'
    return json.dumps(_key_part(part))


def field_name(collection: str, key: Hashable) -> str:
    """Name the field of one entry: its collection, then its key's components as a
    compact JSON array, so distinct keys of distinct types never share a field."""
    if not isinstance(key, tuple):
        return f"{collection}:[{_encoded_part(key)}]"
    return f"{collection}:[{','.join([_encoded_part(p) for p in key])}]"


def keyed_value(ordinal: int, value: BaseModel | str) -> str:
    encoded = (
        value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value)
    )
    return f"{ordinal}:{encoded}"


def scalar_field(name: str) -> str:
    return f"{META}:{name}"


def foundation_fields(snapshot_fields: Mapping[str, Any]) -> dict[str, str]:
    """Encode the fields of a ledger's layout and foundation."""
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
    """Encode every field of a ledger."""
    fields = foundation_fields(foundation)
    for collection, (entries, ordinals) in keyed.items():
        for key, value in entries.items():
            fields[field_name(collection, key)] = keyed_value(ordinals[key], value)
    for collection, history in histories.items():
        prefix = f"{collection}:["
        for position, entry in enumerate(history):
            fields[f"{prefix}{position}]"] = entry.model_dump_json()
    for collection, members in sets.items():
        for member in members:
            fields[field_name(collection, member)] = SET_MEMBER
    for name, value in scalars.items():
        fields[scalar_field(name)] = scalar_value(value)
    return fields


def encode_ledger(stored: StoredLedger) -> dict[str, str]:
    """Encode every field of a stored ledger."""
    snapshot, ordinals = stored.snapshot, stored.ordinals

    def keyed(collection: str) -> KeyedEntries:
        order = ordinals.get(collection, {})
        if collection in KEYED_TEXTS:
            return getattr(snapshot, collection), order
        key_of = KEYED[collection][1]
        entities: list[BaseModel] = getattr(snapshot, collection)
        return {key_of(e): e for e in entities}, order

    return encode_fields(
        {name: getattr(snapshot, name) for name in FOUNDATION},
        {collection: keyed(collection) for collection in (*KEYED, *KEYED_TEXTS)},
        {collection: getattr(snapshot, collection) for collection in HISTORIES},
        {collection: getattr(snapshot, collection) for collection in SETS},
        {name: getattr(snapshot, name) for name in SCALARS},
    )


class LedgerLayoutError(ValueError):
    """A stored ledger holds a field this layout cannot place."""


@cache
def _entity_list(model: type[BaseModel]) -> TypeAdapter[list[Any]]:
    return TypeAdapter(GenericAlias(list, (model,)))


def _validated(model: type[BaseModel], bodies: list[str]) -> list[Any]:
    """Validate one collection's entries in one pass, so its entities are allocated
    together rather than between the decoder's temporaries."""
    entities = _entity_list(model).validate_json("[" + ",".join(bodies) + "]")
    if len(entities) != len(bodies):
        raise LedgerLayoutError(f"an entry of {model.__name__} holds several")
    return entities


def _position(encoded: str) -> int | None:
    digits = encoded[1:-1]
    if (
        encoded[:1] == "["
        and encoded[-1:] == "]"
        and digits.isascii()
        and digits.isdigit()
        and (digits == "0" or digits[0] != "0")
    ):
        return int(digits)
    return None


def _member(collection: str, name: str) -> str:
    """The string key a member or string field is named by, refusing any other
    spelling of it."""
    parts = json.loads(name.partition(":")[2])
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], str)
        or field_name(collection, parts[0]) != name
    ):
        raise LedgerLayoutError(f"malformed key {name!r}")
    return parts[0]


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
    # Fields by collection as parallel lists of names and values, so grouping
    # allocates no object per field for the collector to track.
    names_of: dict[str, list[str]] = {}
    values_of: dict[str, list[str]] = {}
    for name, value in fields.items():
        collection, sep, _ = name.partition(":")
        if not sep:
            raise LedgerLayoutError(f"malformed field {name!r}")
        if (names := names_of.get(collection)) is None:
            names = names_of[collection] = []
            values_of[collection] = []
        names.append(name)
        values_of[collection].append(value)

    scalars: dict[str, Any] = {}
    ordinals: dict[str, dict[Hashable, int]] = {}
    for collection, names in names_of.items():
        values = values_of[collection]
        if collection == META:
            for name, value in zip(names, values):
                if (scalar := name[len(META) + 1 :]) in SCALARS:
                    scalars[scalar] = json.loads(value)
                elif name not in (_LAYOUT_FIELD, _FOUNDATION_FIELD):
                    raise LedgerLayoutError(f"unknown ledger field {name!r}")
        elif collection in KEYED or collection in KEYED_TEXTS:
            data[collection], ordinals[collection] = _keyed_entries(
                collection, names, values
            )
        elif collection in HISTORIES:
            data[collection] = _validated(
                HISTORIES[collection], _history_entries(collection, names, values)
            )
        elif collection in SETS:
            if any(value != SET_MEMBER for value in values):
                raise LedgerLayoutError(f"malformed member of {collection}")
            data[collection] = sorted(_member(collection, name) for name in names)
        else:
            raise LedgerLayoutError(f"unknown ledger collection {collection!r}")
    if scalars.keys() != set(SCALARS):
        raise LedgerLayoutError("ledger does not hold every scalar")
    data.update(scalars)
    return StoredLedger(LedgerSnapshot.model_validate(data), ordinals)


def _keyed_entries(
    collection: str, names: list[str], values: list[str]
) -> tuple[Any, dict[Hashable, int]]:
    """A keyed collection's entries in insertion order, with each key's ordinal."""
    stored = [0] * len(values)
    bodies = [""] * len(values)
    for i, value in enumerate(values):
        ordinal, sep, bodies[i] = value.partition(":")
        if not sep or not ordinal.isascii() or not ordinal.isdigit():
            raise LedgerLayoutError(f"entry of {collection} has no ordinal")
        stored[i] = int(ordinal)
    if len(set(stored)) != len(stored):
        raise LedgerLayoutError(f"conflicting ordinals in {collection}")
    order = sorted(range(len(stored)), key=stored.__getitem__)
    if collection in KEYED_TEXTS:
        keys = [_member(collection, names[i]) for i in order]
        return (
            {key: json.loads(bodies[i]) for key, i in zip(keys, order)},
            {key: stored[i] for key, i in zip(keys, order)},
        )
    model, key_of = KEYED[collection]
    entities = _validated(model, [bodies[i] for i in order])
    ordinals: dict[Hashable, int] = {}
    for i, entity in zip(order, entities):
        key = key_of(entity)
        if field_name(collection, key) != names[i]:
            raise LedgerLayoutError(f"{names[i]!r} holds the entry of {key!r}")
        ordinals[key] = stored[i]
    return entities, ordinals


def _history_entries(collection: str, names: list[str], values: list[str]) -> list[str]:
    """A history's entries by position; each position must be named exactly as the
    writer names it, so the positions cover the history only if none is missing."""
    entries: list[str | None] = [None] * len(values)
    prefix = len(collection) + 1
    for name, value in zip(names, values):
        if (position := _position(name[prefix:])) is None:
            raise LedgerLayoutError(f"malformed position {name!r}")
        if position >= len(entries):
            raise LedgerLayoutError(f"{collection} is missing an entry")
        entries[position] = value
    return cast(list[str], entries)

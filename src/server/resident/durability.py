"""The durable layout of the resident-capacity control facts and its store contract.

The live facts — families, replica incarnations, allocation leases, invocation requests
and claims — persist as one hash with a field per entity, so a step writes only the
entities it changed and a restart loads only what is live. A retired invocation leaves a
compact ``RetiredInvocation`` record, and a retired replica or serve-task family moves
to stored history; neither is loaded at startup.

Retiring an invocation and archiving a replica or family each run as one atomic step.
It refuses, before writing anything, a key of the wrong type or stored state that does
not support the move: an unconsumed request, an open claim, a stored record with a
higher epoch, an invocation with neither live fields nor a record, or a replica stored
live and not inert. It then writes the record or history before deleting the live
fields it replaces, so a failure at any point leaves the evidence in one place or both,
and a retry after a lost reply converges on the same result.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .state import (
    INERT_REPLICA_STATES,
    ClaimState,
    ResidentSnapshot,
    RetiredInvocation,
    ServiceFamily,
    StoredReplica,
)

FAMILY = "family"
REPLICA = "replica"
LEASE = "lease"
REQUEST = "request"
CLAIM = "claim"

# The snapshot list each live field kind decodes into, in load order.
_SNAPSHOT_LISTS = {
    FAMILY: "families",
    REPLICA: "replicas",
    LEASE: "leases",
    REQUEST: "invocations",
    CLAIM: "claims",
}


class ResidentStateError(RuntimeError):
    """The stored resident state is unreadable, inconsistent or refuses a write."""


def live_field(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def decode_live(fields: Mapping[str, str]) -> ResidentSnapshot:
    """Decode the live hash into its facts, refusing any inconsistency.

    A family definition that fails validation is dropped, as the snapshot does; any
    other unreadable field, a claim whose request is missing, or a consumed request with
    an open claim refuses the whole load.
    """
    lists: dict[str, list[Any]] = {name: [] for name in _SNAPSHOT_LISTS.values()}
    for field, value in fields.items():
        kind, sep, key = field.partition(":")
        if not sep or kind not in _SNAPSHOT_LISTS:
            raise ResidentStateError(f"unknown resident field {field!r}")
        try:
            entry = json.loads(value)
        except ValueError as exc:
            raise ResidentStateError(f"unreadable resident field {field!r}") from exc
        if not isinstance(entry, dict):
            raise ResidentStateError(f"unreadable resident field {field!r}")
        lists[_SNAPSHOT_LISTS[kind]].append(entry)
    try:
        snapshot = ResidentSnapshot.model_validate(lists)
    except ValidationError as exc:
        raise ResidentStateError(f"unreadable resident state: {exc}") from exc
    by_invocation = {request.invocation_id: request for request in snapshot.invocations}
    for claim in snapshot.claims:
        if (request := by_invocation.get(claim.invocation_id)) is None:
            raise ResidentStateError(f"claim {claim.claim_id} names no stored request")
        if request.consumed is not None and claim.state is not ClaimState.TERMINAL:
            raise ResidentStateError(
                f"claim {claim.claim_id} is open on a consumed invocation"
            )
    return snapshot


def decode_retired(invocation_id: str, value: str) -> RetiredInvocation:
    """Decode a stored retirement record, refusing one naming another invocation."""
    try:
        record = RetiredInvocation.model_validate_json(value)
    except ValidationError as exc:
        raise ResidentStateError(
            f"unreadable retirement record for {invocation_id}"
        ) from exc
    if record.invocation_id != invocation_id:
        raise ResidentStateError(f"retirement record mismatch for {invocation_id}")
    return record


def decode_stored[M: BaseModel](model: type[M], value: str) -> M:
    """Decode one stored history entry."""
    try:
        return model.model_validate_json(value)
    except ValidationError as exc:
        raise ResidentStateError(f"unreadable stored {model.__name__}") from exc


def check_retirement(
    live: Mapping[str, str],
    retired: str | None,
    record: RetiredInvocation,
    fields: Sequence[str],
) -> None:
    """Refuse a retirement the stored state does not support, before any write.

    ``fields`` are the invocation's live fields, its request's first.
    """
    request = live.get(fields[0])
    if request is not None and json.loads(request).get("consumed") is None:
        raise ResidentStateError(f"invocation {record.invocation_id} is not consumed")
    for field in fields[1:]:
        if (claim := live.get(field)) is not None and json.loads(claim).get(
            "state"
        ) != ClaimState.TERMINAL:
            raise ResidentStateError(f"{field} is not terminal")
    if retired is None:
        if request is None:
            raise ResidentStateError(
                f"invocation {record.invocation_id} has no evidence to retire"
            )
    elif decode_retired(record.invocation_id, retired).epoch_high > record.epoch_high:
        raise ResidentStateError(
            f"invocation {record.invocation_id} has a higher stored epoch"
        )


def check_replica_archive(live: Mapping[str, str], fields: Sequence[str]) -> None:
    """Refuse to archive a replica, the first of ``fields``, stored live and active."""
    if (replica := live.get(fields[0])) is not None and json.loads(replica).get(
        "state"
    ) not in INERT_REPLICA_STATES:
        raise ResidentStateError(f"{fields[0]} is not inert")


class ResidentDurability(Protocol):
    """Where the resident-capacity control facts are made durable.

    Every method raises on a store failure, a refused write or an unreadable entry.
    """

    def write(self, changes: Mapping[str, str | None]) -> None:
        """Set each live field to its value, or delete it for ``None``, atomically."""
        ...

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        """Publish an invocation's retirement record, then delete its live fields."""
        ...

    def retired_invocations(
        self, invocation_ids: Sequence[str]
    ) -> list[RetiredInvocation | None]:
        """Read each invocation's stored retirement record, ``None`` for none."""
        ...

    def archive_replica(self, stored: StoredReplica, fields: Sequence[str]) -> None:
        """Store an inert replica's history, then delete its live fields."""
        ...

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        """Store a retired family's definition, then delete its live field."""
        ...

    def stored_family(self, family: str) -> ServiceFamily | None:
        """Read one retired family's stored definition, ``None`` when it has none."""
        ...

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        """Load the live facts, ``None`` when nothing is stored."""
        ...

    async def stored_replicas_async(self) -> list[StoredReplica]:
        """Read every retired replica incarnation, in no particular order."""
        ...

    async def stored_families_async(self) -> list[ServiceFamily]:
        """Read every retired family definition, in no particular order."""
        ...


class MemoryResidentDurability:
    """Keeps the durable layout in process, encoded as the store keeps it."""

    def __init__(self) -> None:
        self.live: dict[str, str] = {}
        self.retired: dict[str, str] = {}
        self.replicas: dict[str, str] = {}
        self.families: dict[str, str] = {}

    def write(self, changes: Mapping[str, str | None]) -> None:
        for field, value in changes.items():
            if value is None:
                self.live.pop(field, None)
            else:
                self.live[field] = value

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        check_retirement(
            self.live, self.retired.get(record.invocation_id), record, fields
        )
        self.retired[record.invocation_id] = record.model_dump_json()
        for field in fields:
            self.live.pop(field, None)

    def retired_invocations(
        self, invocation_ids: Sequence[str]
    ) -> list[RetiredInvocation | None]:
        return [
            (
                decode_retired(invocation_id, value)
                if (value := self.retired.get(invocation_id)) is not None
                else None
            )
            for invocation_id in invocation_ids
        ]

    def archive_replica(self, stored: StoredReplica, fields: Sequence[str]) -> None:
        check_replica_archive(self.live, fields)
        self.replicas[stored.replica.replica_id] = stored.model_dump_json()
        for field in fields:
            self.live.pop(field, None)

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        self.families[family.family] = family.model_dump_json()
        self.live.pop(field, None)

    def stored_family(self, family: str) -> ServiceFamily | None:
        value = self.families.get(family)
        return decode_stored(ServiceFamily, value) if value is not None else None

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        return decode_live(self.live) if self.live else None

    async def stored_replicas_async(self) -> list[StoredReplica]:
        return [decode_stored(StoredReplica, v) for v in self.replicas.values()]

    async def stored_families_async(self) -> list[ServiceFamily]:
        return [decode_stored(ServiceFamily, v) for v in self.families.values()]

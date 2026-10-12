"""The in-process durability and the Redis registry keep the same layout and refuse
the same retirements, before writing anything."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import fakeredis
import pytest

from server.clients.redis import (
    resident_cs_key,
    resident_family_history_key,
    resident_replica_history_key,
    resident_retired_key,
)
from server.registries.resident import ResidentRegistry
from server.resident import (
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaIncarnation,
    ReplicaState,
    RetiredInvocation,
    ServiceFamily,
    StoredReplica,
)
from server.resident.durability import MemoryResidentDurability, ResidentStateError
from tests.server.redis_helpers import fake_redis_client, raw_control

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")
_REQUEST = "request:inv-1"
_CLAIM = "claim:scl-1"


class _Store:
    """One durability and raw access to the hashes it keeps."""

    def __init__(self, durability: Any, hashes: Callable[[str], dict[str, str]]):
        self.durability = durability
        self._hashes = hashes

    def hash(self, name: str) -> dict[str, str]:
        return self._hashes(name)

    def seed(self, name: str, field: str, value: str) -> None:
        self._hashes(name)[field] = value

    def snapshot(self) -> dict[str, dict[str, str]]:
        return {name: dict(self.hash(name)) for name in _HASHES}


_HASHES = ("live", "retired", "replicas", "families")


def _memory() -> _Store:
    durability = MemoryResidentDurability()
    return _Store(durability, lambda name: getattr(durability, name))


class _RedisHash(dict[str, str]):
    def __init__(self, raw: Any, key: str) -> None:
        super().__init__(raw.hgetall(key))
        self._raw, self._key = raw, key

    def __setitem__(self, field: str, value: str) -> None:
        super().__setitem__(field, value)
        self._raw.hset(self._key, field, value)


def _redis() -> _Store:
    registry = ResidentRegistry(fake_redis_client(fakeredis.FakeServer()))
    raw = raw_control(registry)
    keys = {
        "live": resident_cs_key(),
        "retired": resident_retired_key(),
        "replicas": resident_replica_history_key(),
        "families": resident_family_history_key(),
    }
    return _Store(registry, lambda name: _RedisHash(raw, keys[name]))


_STORES = pytest.mark.parametrize("make", [_memory, _redis], ids=["memory", "redis"])


def _request(consumed: str | None) -> str:
    return json.dumps(
        {
            "invocation_id": "inv-1",
            "subject": _SUBJECT.model_dump(mode="json"),
            "family": "fam",
            "profile": {"engine_batch_key": "fam"},
            "consumed": consumed,
        }
    )


def _claim(state: str) -> str:
    return json.dumps({"claim_id": "scl-1", "invocation_id": "inv-1", "state": state})


def _record(epoch_high: int = 0) -> RetiredInvocation:
    return RetiredInvocation(
        invocation_id="inv-1",
        subject=_SUBJECT,
        family="fam",
        epoch_high=epoch_high,
        terminal_reason=ClaimTerminalReason.COMPLETED,
    )


def _replica(state: ReplicaState) -> ReplicaIncarnation:
    return ReplicaIncarnation(
        replica_id="rpl-1", family="fam", incarnation=1, state=state
    )


_FAMILY = ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")


@_STORES
@pytest.mark.parametrize(
    "live, retired, epoch",
    [
        ({_REQUEST: _request(None)}, None, 0),
        ({_REQUEST: _request("completed"), _CLAIM: _claim("accepted")}, None, 0),
        ({}, None, 0),
        ({_REQUEST: _request("completed")}, _record(5), 0),
    ],
    ids=["unconsumed", "open-claim", "no-evidence", "higher-epoch"],
)
def test_an_unsupported_retirement_writes_nothing(
    make: Callable[[], _Store],
    live: dict[str, str],
    retired: RetiredInvocation | None,
    epoch: int,
) -> None:
    store = make()
    for field, value in live.items():
        store.seed("live", field, value)
    if retired is not None:
        store.seed("retired", "inv-1", retired.model_dump_json())
    before = store.snapshot()

    with pytest.raises(ResidentStateError):
        store.durability.retire_invocation(_record(epoch), [_REQUEST, _CLAIM])

    assert store.snapshot() == before


@_STORES
def test_a_retirement_publishes_its_record_and_repeats_to_the_same_result(
    make: Callable[[], _Store],
) -> None:
    store = make()
    store.seed("live", _REQUEST, _request("completed"))
    store.seed("live", _CLAIM, _claim("terminal"))

    record = _record()
    store.durability.retire_invocation(record, [_REQUEST, _CLAIM])
    after = store.snapshot()
    store.durability.retire_invocation(record, [_REQUEST, _CLAIM])

    assert after["live"] == {}
    assert store.durability.retired_invocations(["inv-1", "inv-9"]) == [
        RetiredInvocation.model_validate_json(after["retired"]["inv-1"]),
        None,
    ]
    assert store.snapshot() == after


@_STORES
def test_a_replica_stored_live_and_active_is_never_archived(
    make: Callable[[], _Store],
) -> None:
    store = make()
    store.seed("live", "replica:rpl-1", _replica(ReplicaState.WARM).model_dump_json())
    before = store.snapshot()

    with pytest.raises(ResidentStateError):
        store.durability.archive_replica(
            StoredReplica(replica=_replica(ReplicaState.STOPPED)), ["replica:rpl-1"]
        )

    assert store.snapshot() == before


@_STORES
def test_an_inert_replica_and_a_family_move_to_their_history(
    make: Callable[[], _Store],
) -> None:
    store = make()
    stopped = _replica(ReplicaState.STOPPED)
    store.seed("live", "replica:rpl-1", stopped.model_dump_json())
    store.seed("live", "family:fam", _FAMILY.model_dump_json())

    store.durability.archive_replica(StoredReplica(replica=stopped), ["replica:rpl-1"])
    store.durability.archive_family(_FAMILY, "family:fam")

    assert store.hash("live") == {}
    assert asyncio.run(store.durability.stored_replicas_async()) == [
        StoredReplica(replica=stopped)
    ]
    assert asyncio.run(store.durability.stored_families_async()) == [_FAMILY]
    assert store.durability.stored_family("fam") == _FAMILY
    assert store.durability.stored_family("other") is None


@_STORES
@pytest.mark.parametrize("history", ["replicas", "families"])
def test_an_unreadable_history_entry_raises_a_state_error(
    make: Callable[[], _Store], history: str
) -> None:
    store = make()
    store.seed(history, "x", "not json")
    read = (
        store.durability.stored_replicas_async
        if history == "replicas"
        else store.durability.stored_families_async
    )

    with pytest.raises(ResidentStateError):
        asyncio.run(read())

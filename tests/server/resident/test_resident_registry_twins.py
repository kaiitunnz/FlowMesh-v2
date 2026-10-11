"""The resident registry's sync and async twins write and read alike."""

import asyncio
from collections.abc import Callable
from typing import Any

import fakeredis
import pytest

from server.clients.redis import (
    resident_cs_key,
    resident_family_history_key,
    resident_replica_history_key,
    resident_retired_key,
    serve_cs_key,
)
from server.registries.resident import ResidentRegistry
from server.resident import (
    AdmissionController,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaState,
    RetiredInvocation,
    ServiceFamily,
    StoredReplica,
)
from server.serve import ServeSnapshot
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.resident._helpers import PROFILE, warm_stores

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")
_KEYS = (
    resident_cs_key(),
    resident_retired_key(),
    resident_replica_history_key(),
    resident_family_history_key(),
    serve_cs_key(),
)


def _seeded() -> ResidentRegistry:
    """A store holding a consumed invocation and a stopped replica, all durable."""
    registry = ResidentRegistry(fake_redis_client(fakeredis.FakeServer()))
    stores = warm_stores(durability=registry)
    admission = AdmissionController(stores)
    claim = admission.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="fam", profile=PROFILE
    )
    admission.admit(claim, PROFILE, idempotency_key=None)
    admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    replica = stores.directory.get("rpl-1")
    assert replica is not None
    replica.state = ReplicaState.STOPPED
    stores.flush()
    return registry


def _copy(registry: ResidentRegistry) -> ResidentRegistry:
    """A second store holding exactly what ``registry`` holds."""
    copy = ResidentRegistry(fake_redis_client(fakeredis.FakeServer()))
    raw, target = raw_control(registry), raw_control(copy)
    for key in _KEYS:
        if raw.type(key) == "hash":
            target.hset(key, mapping=raw.hgetall(key))
    return copy


def _stored(registry: ResidentRegistry) -> dict[str, Any]:
    raw = raw_control(registry)
    return {
        key: raw.hgetall(key) if raw.type(key) == "hash" else raw.get(key)
        for key in _KEYS
    }


_RECORD = RetiredInvocation(
    invocation_id="inv-1",
    subject=_SUBJECT,
    family="fam",
    epoch_high=0,
    terminal_reason=ClaimTerminalReason.COMPLETED,
)
_FAMILY = ServiceFamily(family="fam", engine_batch_key="fam", model_ref="m")


def _replica(registry: ResidentRegistry) -> StoredReplica:
    snapshot = registry.load_snapshot()
    assert snapshot is not None
    [replica] = snapshot.replicas
    return StoredReplica(replica=replica)


# Each write: (sync name, its arguments from the seeded registry).
_WRITES: dict[str, Callable[[ResidentRegistry], tuple[Any, ...]]] = {
    "write": lambda _r: ({"family:extra": _FAMILY.model_dump_json(), "x": None},),
    "retire_invocation": lambda r: (
        _RECORD,
        [
            field
            for field in raw_control(r).hkeys(resident_cs_key())
            if field.startswith(("request:", "claim:"))
        ],
    ),
    "archive_replica": lambda r: (_replica(r), ["replica:rpl-1"]),
    "archive_family": lambda _r: (_FAMILY, "family:fam"),
    "save_serve_snapshot": lambda _r: (ServeSnapshot(),),
}


@pytest.mark.parametrize("name", list(_WRITES))
def test_each_write_stores_alike_sync_and_async(name: str) -> None:
    sync_registry = _seeded()
    async_registry = _copy(sync_registry)
    args = _WRITES[name](sync_registry)

    getattr(sync_registry, name)(*args)
    asyncio.run(getattr(async_registry, f"{name}_async")(*args))

    assert _stored(sync_registry) == _stored(async_registry)


_READS: dict[str, tuple[Any, ...]] = {
    "retired_invocations": (["inv-1", "inv-9"],),
    "stored_family": ("fam",),
    "load_snapshot": (),
    "stored_replicas": (),
    "stored_families": (),
    "load_serve_snapshot": (),
}


@pytest.mark.parametrize("name", list(_READS))
def test_each_read_answers_alike_sync_and_async(name: str) -> None:
    registry = _seeded()
    registry.retire_invocation(*_WRITES["retire_invocation"](registry))
    registry.archive_replica(*_WRITES["archive_replica"](registry))
    registry.archive_family(_FAMILY, "family:fam")
    registry.save_serve_snapshot(ServeSnapshot())
    args = _READS[name]

    assert getattr(registry, name)(*args) == asyncio.run(
        getattr(registry, f"{name}_async")(*args)
    )


def test_every_public_method_has_its_twin() -> None:
    public = {
        name
        for name in vars(ResidentRegistry)
        if not name.startswith("_") and callable(getattr(ResidentRegistry, name))
    }
    sync = {name for name in public if not name.endswith("_async")}

    assert {f"{name}_async" for name in sync} == public - sync

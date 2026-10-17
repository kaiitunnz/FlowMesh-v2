"""Assertion holds: who may hold an identity, and what refuses or releases a hold."""

import json
from typing import cast

import fakeredis
import pytest

from server.clients.redis import (
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
)
from server.reclamation.pins import CHUNK, AcquireMode, PinStore, Verdict
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"
_LOCATOR = {"kind": "result", "task_id": "tsk-1"}


@pytest.fixture
def store() -> tuple[PinStore, fakeredis.FakeRedis]:
    server = fakeredis.FakeServer()
    client = fake_redis_client(server)
    return PinStore(client), fakeredis.FakeRedis(server=server, decode_responses=True)


def _holders(raw: fakeredis.FakeRedis, digest: str) -> list[str]:
    value = cast(str | None, raw.hget(reclaim_pins_key(_SCOPE), digest))
    return [holder["n"] for holder in json.loads(value)] if value else []


def test_an_acquisition_holds_unprotected_identities_and_cancels_condemnation(
    store,
) -> None:
    pins, raw = store
    raw.sadd(reclaim_protected_key(_SCOPE), "p")
    raw.hset(reclaim_condemned_key(_SCOPE), "x", "{}")
    result = pins.acquire(_SCOPE, ["x", "p"], "1:1", _LOCATOR)
    assert result.verdict is Verdict.ACQUIRED
    assert result.held == ("x",)
    assert _holders(raw, "x") == ["1:1"]
    assert not raw.hexists(reclaim_condemned_key(_SCOPE), "x")
    assert _holders(raw, "p") == []


def test_retirement_refuses_before_the_protected_fast_path(store) -> None:
    pins, raw = store
    raw.sadd(reclaim_protected_key(_SCOPE), "x")
    raw.hset(reclaim_deleting_key(_SCOPE), "x", "{}")
    result = pins.acquire(_SCOPE, ["x"], "1:1", _LOCATOR)
    assert result.verdict is Verdict.RETIRING
    assert result.retiring == ("x",)


def test_a_refused_chunk_holds_nothing(store) -> None:
    pins, raw = store
    raw.hset(reclaim_deleting_key(_SCOPE), "b", "{}")
    raw.hset(reclaim_condemned_key(_SCOPE), "a", "{}")
    result = pins.acquire(_SCOPE, ["a", "b"], "1:1", _LOCATOR, manifests=["m"])
    assert result.verdict is Verdict.RETIRING
    assert _holders(raw, "a") == []
    assert raw.hexists(reclaim_condemned_key(_SCOPE), "a")
    assert not raw.sismember(reclaim_unexpanded_key(_SCOPE), "m")


def test_a_refusal_in_a_later_chunk_releases_the_earlier_chunks(store) -> None:
    pins, raw = store
    digests = [f"d{i}" for i in range(CHUNK + 1)]
    raw.hset(reclaim_deleting_key(_SCOPE), digests[-1], "{}")
    result = pins.acquire(_SCOPE, digests, "1:1", _LOCATOR)
    assert result.verdict is Verdict.RETIRING
    assert raw.hlen(reclaim_pins_key(_SCOPE)) == 0


def test_a_retired_identity_needs_its_backing_verified_at_its_revision(store) -> None:
    pins, raw = store
    raw.hset(reclaim_history_key(_SCOPE), "x", "3")
    first = pins.acquire(_SCOPE, ["x"], "1:1", _LOCATOR)
    assert first.verdict is Verdict.NEEDS_VERIFY
    assert first.unverified == {"x": "3"}
    stale = pins.acquire(_SCOPE, ["x"], "1:2", _LOCATOR, verified={"x": "2"})
    assert stale.verdict is Verdict.NEEDS_VERIFY
    verified = pins.acquire(_SCOPE, ["x"], "1:3", _LOCATOR, verified={"x": "3"})
    assert verified.verdict is Verdict.ACQUIRED


def test_a_pending_acquisition_holds_through_a_retirement(store) -> None:
    pins, raw = store
    raw.hset(reclaim_deleting_key(_SCOPE), "x", "{}")
    raw.hset(reclaim_history_key(_SCOPE), "x", "1")
    result = pins.acquire(_SCOPE, ["x"], "1:1", _LOCATOR, AcquireMode.PENDING)
    assert result.verdict is Verdict.ACQUIRED
    assert _holders(raw, "x") == ["1:1"]


def test_holders_never_overwrite_or_release_each_other(store) -> None:
    pins, raw = store
    pins.acquire(_SCOPE, ["x"], "1:1", _LOCATOR)
    pins.acquire(_SCOPE, ["x"], "1:2", _LOCATOR)
    assert _holders(raw, "x") == ["1:1", "1:2"]
    assert pins.release(_SCOPE, ["x"], "1:1") == 1
    assert _holders(raw, "x") == ["1:2"]
    assert pins.release(_SCOPE, ["x"], "1:1") == 0
    assert pins.release(_SCOPE, ["x"], "1:2") == 1
    assert not raw.hexists(reclaim_pins_key(_SCOPE), "x")


def test_an_acquisition_refuses_a_key_of_the_wrong_type(store) -> None:
    pins, raw = store
    raw.set(reclaim_pins_key(_SCOPE), "corrupt")
    with pytest.raises(Exception, match="WRONGTYPE"):
        pins.acquire(_SCOPE, ["x"], "1:1", _LOCATOR)
    assert raw.get(reclaim_pins_key(_SCOPE)) == "corrupt"


def test_a_new_root_releases_only_the_holds_earlier_roots_took(store) -> None:
    pins, raw = store
    raw.sadd("rc:scopes", _SCOPE)
    pins.acquire(_SCOPE, ["a", "b"], "3:1", _LOCATOR)
    pins.acquire(_SCOPE, ["b"], "4:1", _LOCATOR)
    pins.acquire(_SCOPE, ["c"], "5:2", _LOCATOR)
    assert pins.release_older(5, page=1) == 3
    assert _holders(raw, "a") == []
    assert _holders(raw, "b") == []
    assert _holders(raw, "c") == ["5:2"]

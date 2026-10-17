"""Guarded commits: a write carrying asserted state lands only under the root epoch
that issued it, and a lost watch is told apart from a later root."""

import asyncio

import fakeredis
import pytest
from redis.exceptions import WatchError

from server.clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    reclaim_protected_key,
    workflow_ds_key,
)
from server.orchestration.ledger_layout import LedgerChanges
from server.reclamation.epoch import CommitGuard, EpochFenced, GuardInterrupted
from server.registries.workflow import WorkflowRegistry, _guard_sync
from server.task.runtime.commits import store_unavailable
from shared.content import reference_for
from tests.server.redis_helpers import fake_redis_client

_WF = "wfl-1"
_REF = reference_for("org-a", b"bytes")


def _changes() -> LedgerChanges:
    return LedgerChanges({"f": "1:{}"}, references=(_REF,))


@pytest.fixture
def setup() -> tuple[WorkflowRegistry, fakeredis.FakeRedis]:
    server = fakeredis.FakeServer()
    raw = fakeredis.FakeRedis(server=server, decode_responses=True)
    return WorkflowRegistry(fake_redis_client(server)), raw


def test_a_guarded_commit_lands_under_its_epoch_with_its_protection(setup) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()
    registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert raw.hget(workflow_ds_key(_WF), "f") == "1:{}"
    assert raw.sismember(reclaim_protected_key("org-a"), _REF.content_digest)


def test_a_later_root_fences_a_commit_before_it_starts(setup) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()
    registry.claim_root_epoch()
    with pytest.raises(EpochFenced):
        registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert not raw.exists(workflow_ds_key(_WF))
    assert not raw.exists(reclaim_protected_key("org-a"))


def test_a_later_root_fences_a_commit_already_issued(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()

    def later_root_starts(pipe, guard):
        _guard_sync(pipe, guard)
        raw.incr(RECLAIM_ROOT_EPOCH_KEY)

    monkeypatch.setattr("server.registries.workflow._guard_sync", later_root_starts)
    with pytest.raises(EpochFenced):
        registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert not raw.exists(workflow_ds_key(_WF))
    assert not raw.exists(reclaim_protected_key("org-a"))


def test_a_lost_watch_under_the_same_epoch_is_retriable(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()

    def watch_fires(pipe, guard):
        _guard_sync(pipe, guard)
        raw.set(RECLAIM_ROOT_EPOCH_KEY, str(epoch))

    monkeypatch.setattr("server.registries.workflow._guard_sync", watch_fires)
    with pytest.raises(GuardInterrupted) as caught:
        registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert store_unavailable(caught.value)
    assert not raw.exists(workflow_ds_key(_WF))


def test_an_unguarded_commit_ignores_the_epoch(setup) -> None:
    registry, raw = setup
    registry.claim_root_epoch()
    registry.save_ledger(_WF, _changes())
    assert raw.hget(workflow_ds_key(_WF), "f") == "1:{}"


def test_the_async_guard_fences_a_later_root(setup) -> None:
    registry, raw = setup

    async def run() -> None:
        epoch = await registry.claim_root_epoch_async()
        await registry.save_ledger_async(_WF, _changes(), guard=CommitGuard(epoch))
        await registry.claim_root_epoch_async()
        with pytest.raises(EpochFenced):
            await registry.save_ledger_async(
                _WF, LedgerChanges({"g": "2:{}"}), guard=CommitGuard(epoch)
            )

    asyncio.run(run())
    assert raw.hget(workflow_ds_key(_WF), "f") == "1:{}"
    assert not raw.hexists(workflow_ds_key(_WF), "g")


def test_a_fenced_reset_leaves_the_stored_ledger_whole(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()
    registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))

    def later_root_starts(pipe, guard):
        _guard_sync(pipe, guard)
        raw.incr(RECLAIM_ROOT_EPOCH_KEY)

    monkeypatch.setattr("server.registries.workflow._guard_sync", later_root_starts)
    reset = LedgerChanges(
        {"g": "2:{}"}, references=(reference_for("org-a", b"other"),), reset=True
    )
    with pytest.raises(EpochFenced):
        registry.save_ledger(_WF, reset, guard=CommitGuard(epoch))
    # Neither the delete nor the rewrite of the reset landed.
    assert raw.hgetall(workflow_ds_key(_WF)) == {"f": "1:{}"}
    assert raw.smembers(reclaim_protected_key("org-a")) == {_REF.content_digest}


def test_a_disconnect_while_the_guard_reads_the_epoch_is_retriable(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, raw = setup
    epoch = registry.claim_root_epoch()

    def disconnects(pipe, guard):
        raise WatchError("A ConnectionError occurred while watching one or more keys")

    monkeypatch.setattr("server.registries.workflow._guard_sync", disconnects)
    with pytest.raises(GuardInterrupted) as caught:
        registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert store_unavailable(caught.value)

    raw.incr(RECLAIM_ROOT_EPOCH_KEY)
    with pytest.raises(EpochFenced):
        registry.save_ledger(_WF, _changes(), guard=CommitGuard(epoch))
    assert not raw.exists(workflow_ds_key(_WF))


def test_an_async_disconnect_while_the_guard_reads_the_epoch_is_retriable(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _ = setup

    async def disconnects(pipe, guard):
        raise WatchError("A ConnectionError occurred while watching one or more keys")

    monkeypatch.setattr("server.registries.workflow._guard_async", disconnects)

    async def run() -> None:
        epoch = await registry.claim_root_epoch_async()
        with pytest.raises(GuardInterrupted):
            await registry.save_ledger_async(_WF, _changes(), guard=CommitGuard(epoch))

    asyncio.run(run())

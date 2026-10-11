"""A workflow's ledger in the control store, written field by field from the changes
each transition made, restores as the workflow ran across every kind of write fault."""

import asyncio
from collections.abc import Callable
from typing import Any, cast

import fakeredis
import pytest
from redis.exceptions import ResponseError

from server.clients.redis import workflow_ds_key
from server.orchestration import OrchestrationEngine
from server.orchestration.ledger_layout import LedgerChanges
from server.registries.workflow import (
    WorkflowControl,
    WorkflowRegistry,
    WorkflowSched,
)
from tests.server.orchestration.helpers import WORKFLOW_ID, chain_bundle, engine
from tests.server.redis_helpers import fake_redis_client
from tests.server.result_store import result_payload
from tests.server.task.test_runtime_control_flow import (
    _CONSUMED,
    _LOOP,
    _Run,
    _workflow,
)
from tests.server.task.test_v2_orchestration import _TS

_LOOP_WORKFLOW = _workflow(_CONSUMED, _LOOP)


class _Store(WorkflowRegistry):
    """The control store, failing ledger writes as ``fault`` names: each refused
    before it runs until cleared, or the next one's reply lost after it applied, or
    the next one faulted by a command the store rejects inside its transaction."""

    def __init__(self) -> None:
        super().__init__(fake_redis_client(fakeredis.FakeServer()))
        self.fault: str | None = None
        self.ledger_writes: list[LedgerChanges] = []

    @property
    def raw(self) -> Any:
        # redis-py types a sync reply as possibly awaitable.
        return self._rds.sync._control

    def _ledger_write(
        self, workflow_id: str, ledger: LedgerChanges, write: Callable[[], None]
    ) -> None:
        fault = self.fault
        if fault != "refused":
            self.fault = None
        if fault == "refused":
            raise ConnectionError("control store unavailable")
        if fault == "faulted":
            self.raw.set(workflow_ds_key(workflow_id), "not a ledger")
        write()
        self.ledger_writes.append(ledger)
        if fault == "applied":
            raise ConnectionError("reply lost after EXEC")

    def save_ledger(
        self,
        workflow_id: str,
        ledger: LedgerChanges,
        control: WorkflowControl | None = None,
    ) -> None:
        self._ledger_write(
            workflow_id,
            ledger,
            lambda: super(_Store, self).save_ledger(workflow_id, ledger, control),
        )

    def commit_dynamic_tasks(
        self,
        workflow_id: str,
        records: Any,
        ledger: LedgerChanges,
        *args: Any,
        **kw: Any,
    ) -> None:
        self._ledger_write(
            workflow_id,
            ledger,
            lambda: super(_Store, self).commit_dynamic_tasks(
                workflow_id, records, ledger, *args, **kw
            ),
        )

    def stored(self, workflow_id: str) -> Any:
        stored = self.load_ledger(workflow_id)
        assert stored is not None
        return stored.snapshot


async def _start(store: _Store) -> _Run:
    return await _Run(cast(Any, store)).start(_LOOP_WORKFLOW)


@pytest.mark.parametrize("fault", ["refused", "applied", "faulted"])
@pytest.mark.anyio
async def test_a_loop_restores_from_its_stored_ledger_after_a_write_fault(
    fault: str,
) -> None:
    store = _Store()
    run = await _start(store)
    run.run("seed")
    run.run("step", {"route": "again"})

    writes = len(store.ledger_writes)
    store.fault = fault
    if fault == "faulted":
        step = next(t for t in run.ready if run.name(t) == "step")
        with pytest.raises(ResponseError, match="WRONGTYPE"):
            run.run("step", {"route": "again"})
        # The worker's report is handled again, as its replay is.
        record = run.runtime.get_record(step)
        assert record is not None
        payload = result_payload(
            run.reader, step, {"ok": True, "route": "again"}, record.org_id
        )
        run.runtime.mark_succeeded(step, "wkr-1", payload, _TS)
        run.drive()
        assert any(w.reset for w in store.ledger_writes[writes:])
    else:
        run.run("step", {"route": "again"})
    if fault == "refused":
        assert run.workflow_id in run.runtime._committer.debt
        assert len(store.ledger_writes) == writes
        store.fault = None
    run.runtime._retry_durability(run.workflow_id)
    assert run.workflow_id not in run.runtime._committer.debt
    assert store.stored(run.workflow_id) == run.engine.to_snapshot()

    restored = await run.restart()
    assert restored.engine.to_snapshot() == run.engine.to_snapshot()
    restored.run("step", {"route": "done"})
    restored.run("consume")
    assert restored.settled()
    assert store.stored(run.workflow_id) == restored.engine.to_snapshot()

    store.unregister_workflows(run.workflow_id)
    assert not store.raw.exists(workflow_ds_key(run.workflow_id))


@pytest.mark.anyio
async def test_a_transition_writes_only_the_ledger_facts_it_changed() -> None:
    store = _Store()
    run = await _start(store)
    run.run("seed")
    for _ in range(12):
        run.run("step", {"route": "again"})
    run.run("step", {"route": "done"})
    run.run("consume")
    assert run.settled()

    writes = store.ledger_writes
    assert not any(w.reset for w in writes)
    assert not any("meta:foundation" in w.fields for w in writes)
    early, late = writes[: len(writes) // 3], writes[-len(writes) // 3 :]
    largest = max(len(w.fields) for w in early)
    assert max(len(w.fields) for w in late) <= largest
    assert store.stored(run.workflow_id) == run.engine.to_snapshot()


_UNREADABLE: dict[str, Callable[[Any, str], Any]] = {
    "malformed field": lambda raw, key: raw.hset(key, "scopes:[bad", "x"),
    "stored as one value": lambda raw, key: (raw.delete(key), raw.set(key, "{}")),
}


@pytest.mark.parametrize("shape", _UNREADABLE)
@pytest.mark.anyio
async def test_an_unreadable_ledger_fails_its_workflow_beside_a_healthy_one(
    shape: str,
) -> None:
    store = _Store()
    broken, healthy = await _start(store), await _start(store)
    broken.run("seed")
    healthy.run("seed")
    _UNREADABLE[shape](store.raw, workflow_ds_key(broken.workflow_id))

    restored = _Run(cast(Any, store), healthy.reader)
    assert await restored.runtime.rehydrate() == 1
    record = store.get_workflow_record(broken.workflow_id)
    assert record is not None
    assert record.control_failure.startswith("UnsupportedWorkflowVersion")
    restored.workflow_id, restored.ids = healthy.workflow_id, healthy.ids
    restored.drive()
    restored.run("step", {"route": "done"})
    restored.run("consume")
    assert restored.settled()


def _written(eng: OrchestrationEngine, store: WorkflowRegistry) -> None:
    changes = eng.ledger_changes()
    store.save_ledger(WORKFLOW_ID, changes)
    eng.ledger_written(changes)


def _restored(eng: OrchestrationEngine, store: WorkflowRegistry) -> OrchestrationEngine:
    stored = store.load_ledger(WORKFLOW_ID)
    assert stored is not None
    return OrchestrationEngine(
        stored.snapshot, eng._topology.bundle, ordinals=stored.ordinals
    )


def test_a_restored_ledger_keeps_its_order_and_owes_nothing_the_store_holds() -> None:
    store = _Store()
    eng = engine(chain_bundle())
    eng.on_dispatched("A", "w1")
    eng.on_failed("A", "boom", retryable=False)
    _written(eng, store)
    reasons = eng._failures.failure_reasons
    assert list(reasons) == ["A", "B"]

    reason = reasons.pop("A")
    _written(eng, store)
    reasons["A"] = reason
    _written(eng, store)

    restored = _restored(eng, store)
    assert list(restored._failures.failure_reasons) == ["B", "A"]
    assert restored.to_snapshot() == eng.to_snapshot()
    owed = restored.ledger_changes()
    assert not owed.reset and not owed.fields and not owed.deleted


def test_a_change_made_while_a_write_is_in_flight_is_not_taken_as_written() -> None:
    store = _Store()
    eng = engine(chain_bundle())
    _written(eng, store)
    eng.on_dispatched("A", "w1")
    changes = eng.ledger_changes()
    store.save_ledger(WORKFLOW_ID, changes)
    eng.on_failed("A", "boom", retryable=False)

    eng.ledger_written(changes)

    _written(eng, store)
    assert _restored(eng, store).to_snapshot() == eng.to_snapshot()


def test_a_rewrite_replaces_whatever_the_store_holds() -> None:
    store = _Store()
    eng = engine(chain_bundle())
    _written(eng, store)
    store.raw.hset(workflow_ds_key(WORKFLOW_ID), "scopes:[bad", "x")

    eng.owe_ledger_rewrite()
    _written(eng, store)

    assert _restored(eng, store).to_snapshot() == eng.to_snapshot()


def test_registration_writes_the_whole_ledger_and_a_restore_reads_it_back() -> None:
    store = _Store()
    eng = engine(chain_bundle())
    changes = eng.ledger_changes()
    assert changes.reset and "meta:foundation" in changes.fields

    asyncio.run(
        store.register_workflow_async(WORKFLOW_ID, [], WorkflowSched(), ledger=changes)
    )
    eng.ledger_written(changes)

    assert _restored(eng, store).to_snapshot() == eng.to_snapshot()
    assert not eng.ledger_changes().fields

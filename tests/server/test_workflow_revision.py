"""Every write of a workflow's stored state bumps its revision, once per write."""

from collections.abc import Callable
from typing import Any

import pytest

from server.clients.redis import (
    WORKFLOWS_OPEN_KEY,
    workflow_revision_key,
    workflow_sources_key,
)
from server.orchestration.ledger_layout import LedgerChanges
from server.registries.workflow import WorkflowControl, WorkflowRegistry
from tests.server.redis_helpers import raw_control
from tests.server.test_workflow_sources import _V1, _register, _registry, _runtime


def _revision(registry: WorkflowRegistry, workflow_id: str) -> int:
    return int(raw_control(registry).get(workflow_revision_key(workflow_id)) or 0)


def _keep_lost_sources(
    registry: WorkflowRegistry, workflow_id: str, items: list[Any]
) -> None:
    raw_control(registry).delete(workflow_sources_key(workflow_id))
    registry.keep_sources(workflow_id, items)


_WRITES: dict[str, Callable[[WorkflowRegistry, str, list[Any]], None]] = {
    "records-only commit": lambda r, w, items: r.commit_transition(w, records=items),
    "membership-only commit": lambda r, w, items: r.commit_transition(
        w, failed=[items[0].record.task_id]
    ),
    "control-only commit": lambda r, w, items: r.commit_transition(
        w, control=WorkflowControl(open=True)
    ),
    "dynamic tasks": lambda r, w, items: r.commit_dynamic_tasks(
        w, items, LedgerChanges(fields={"x": "1"})
    ),
    "standalone task states": lambda r, w, items: r.save_task_states(items),
    "missing sources": lambda r, w, items: _keep_lost_sources(r, w, items),
    "schedule": lambda r, w, items: r.save_workflow_sched(w, True, 1),
    "ledger": lambda r, w, items: r.save_ledger(w, LedgerChanges(fields={"x": "1"})),
}


@pytest.mark.parametrize("write", list(_WRITES))
def test_each_write_of_a_workflow_bumps_its_revision_once(write: str) -> None:
    registry = _registry()
    workflow_id, task_ids = _register(_runtime(registry), _V1)
    items = [item for item in registry.load_task_states(workflow_id, *task_ids) if item]
    before = _revision(registry, workflow_id)
    assert before > 0

    _WRITES[write](registry, workflow_id, items)

    assert _revision(registry, workflow_id) == before + 1


def test_a_registered_workflow_is_open_until_unregistered() -> None:
    registry = _registry()
    workflow_id, _ = _register(_runtime(registry), _V1)
    raw = raw_control(registry)
    assert raw.sismember(WORKFLOWS_OPEN_KEY, workflow_id)

    registry.unregister_workflows(workflow_id)

    assert not raw.sismember(WORKFLOWS_OPEN_KEY, workflow_id)
    assert raw.get(workflow_revision_key(workflow_id)) is None


def test_a_registration_starts_the_revision_at_one() -> None:
    registry = _registry()
    workflow_id, _ = _register(_runtime(registry), _V1)

    assert _revision(registry, workflow_id) == 1

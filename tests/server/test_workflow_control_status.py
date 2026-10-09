"""A workflow's status reads its control summary with every transition that writes
it."""

import fakeredis
import pytest

from server.registries.workflow import (
    WorkflowControl,
    WorkflowRegistry,
    WorkflowSched,
    WorkflowStatus,
)
from tests.server.redis_helpers import fake_redis_client


@pytest.fixture
def registry() -> WorkflowRegistry:
    registry = WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))
    registry.register_workflow("wfl-1", [], WorkflowSched())
    return registry


@pytest.mark.parametrize(
    ("control", "status"),
    [
        (WorkflowControl(), WorkflowStatus.DONE),
        (WorkflowControl(open=True), WorkflowStatus.PENDING),
        (WorkflowControl(cancelled=True), WorkflowStatus.CANCELLED),
        (WorkflowControl(open=True, cancelled=True), WorkflowStatus.PENDING),
        (WorkflowControl(failure="boom", cancelled=True), WorkflowStatus.FAILED),
    ],
)
def test_a_task_commit_writes_the_control_summary_its_status_reads(
    registry: WorkflowRegistry, control: WorkflowControl, status: WorkflowStatus
) -> None:
    registry.commit_transition("wfl-1", control=control)
    workflow = registry.get_workflow("wfl-1")
    assert workflow is not None and workflow.status is status
    record = registry.get_workflow_record("wfl-1")
    assert record is not None
    assert (record.control_open, record.control_cancelled) == (
        control.open,
        control.cancelled,
    )

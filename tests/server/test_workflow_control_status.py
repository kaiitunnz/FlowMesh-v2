"""A workflow's status reads its control summary with every transition that writes
it."""

import logging

import fakeredis
import pytest
from flowmesh.models import Workflow as SdkWorkflow
from lumid_hooks import PrincipalContext

from server.registries.workflow import (
    WorkflowControl,
    WorkflowRegistry,
    WorkflowSched,
    WorkflowStatus,
)
from server.routers.v1 import workflows as workflows_router
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


_BUDGET = "LoopIterationBudgetExceeded: loop refine may run 1 iterations"


@pytest.mark.anyio
async def test_a_control_failure_is_served_as_the_workflows_failure(
    registry: WorkflowRegistry,
) -> None:
    registry.commit_transition("wfl-1", control=WorkflowControl(failure=_BUDGET))

    served = await workflows_router.get_workflow(
        "wfl-1",
        principal=PrincipalContext(
            principal_id="p-1",
            org_id="org",
            external_id="ext",
            principal_type="user",
            scopes=[],
        ),
        registry=registry,
        logger=logging.getLogger("test.workflow_control_status"),
    )
    client = SdkWorkflow.model_validate(served.model_dump(mode="json"))
    assert client.status == "FAILED" and client.failure == _BUDGET


def test_a_workflow_failed_only_by_its_tasks_serves_no_failure(
    registry: WorkflowRegistry,
) -> None:
    registry.commit_transition("wfl-1", failed=["tsk-1"], control=WorkflowControl())
    workflow = registry.get_workflow("wfl-1")
    assert workflow is not None
    assert workflow.status is WorkflowStatus.FAILED and workflow.failure is None

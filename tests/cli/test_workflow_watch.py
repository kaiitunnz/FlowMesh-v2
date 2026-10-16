"""``flowmesh workflow watch``."""

from unittest.mock import MagicMock, patch

from flowmesh.models import Workflow, WorkflowStatus
from flowmesh_cli.cli import build_cli_app
from typer.testing import CliRunner

runner = CliRunner()


def _workflow(status: WorkflowStatus, uploads: int) -> Workflow:
    return Workflow(
        workflow_id="wfl-1",
        task_ids=["tsk-1"],
        submitted_at="2026-10-16T00:00:00+00:00",
        updated_at="2026-10-16T00:00:00+00:00",
        status=status,
        dispatched_tasks=[],
        completed_tasks=["tsk-1"],
        failed_tasks=[],
        cancelled_tasks=[],
        pending_artifact_uploads=uploads,
    )


def test_watch_reports_the_artifact_uploads_a_workflow_waits_on() -> None:
    client = MagicMock()
    client.workflows.retrieve.side_effect = [
        _workflow(WorkflowStatus.PENDING, 2),
        _workflow(WorkflowStatus.PENDING, 1),
        _workflow(WorkflowStatus.DONE, 0),
    ]
    with (
        patch("flowmesh_cli.commands.workflow.FlowMesh", return_value=client),
        patch("flowmesh_cli.commands.workflow.time.sleep"),
    ):
        result = runner.invoke(build_cli_app(), ["workflow", "watch", "wfl-1"])

    assert result.exit_code == 0, result.output
    assert "status: PENDING (2 pending artifact upload(s))" in result.output
    assert "status: PENDING (1 pending artifact upload(s))" in result.output
    assert "status: DONE\n" in result.output

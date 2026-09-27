"""``flowmesh workflow outputs`` / ``workflow output``."""

import json
from unittest.mock import MagicMock, patch

from flowmesh import OutputPendingError
from flowmesh.models import (
    OutputOutcome,
    WorkflowOutputEntry,
    WorkflowOutputPage,
    WorkflowOutputValue,
)
from flowmesh_cli.cli import build_cli_app
from typer.testing import CliRunner

runner = CliRunner()


def _client() -> MagicMock:
    client = MagicMock()
    client.workflows.list_outputs.return_value = WorkflowOutputPage(
        entries=[
            WorkflowOutputEntry(
                cursor="c-1",
                name="fanout",
                cardinality="keyed_collection",
                scope="scp-1",
                key="0",
                outcome=OutputOutcome.EXPLICIT_EMPTY,
            )
        ],
        next_cursor="c-1",
        prev_cursor="c-1",
        open=False,
    )
    client.workflows.get_output.return_value = WorkflowOutputValue(
        name="summarize", cardinality="singleton", outcome=OutputOutcome.SUCCESS
    )
    return client


def test_outputs_pages_by_cursor() -> None:
    client = _client()
    with patch("flowmesh_cli.commands.workflow.FlowMesh", return_value=client):
        result = runner.invoke(
            build_cli_app(),
            ["workflow", "outputs", "wfl-1", "--output", "fanout", "--after", "c-0"],
        )
    assert result.exit_code == 0, result.output
    client.workflows.list_outputs.assert_called_once_with(
        "wfl-1", limit=100, before=None, after="c-0", output="fanout", scope=None
    )
    assert json.loads(result.output)["entries"][0]["outcome"] == "explicit_empty"


def test_output_selects_a_member() -> None:
    client = _client()
    with patch("flowmesh_cli.commands.workflow.FlowMesh", return_value=client):
        result = runner.invoke(
            build_cli_app(),
            ["workflow", "output", "wfl-1", "fanout", "--scope", "scp-1", "--key", "3"],
        )
    assert result.exit_code == 0, result.output
    client.workflows.get_output.assert_called_once_with(
        "wfl-1", "fanout", scope="scp-1", key="3", sequence=None
    )


def test_a_pending_output_exits_with_its_error() -> None:
    client = _client()
    client.workflows.get_output.side_effect = OutputPendingError(
        "not settled", status_code=409, method="GET", url="u"
    )
    with patch("flowmesh_cli.commands.workflow.FlowMesh", return_value=client):
        result = runner.invoke(build_cli_app(), ["workflow", "output", "wfl-1", "s"])
    assert result.exit_code == 1
    assert "409" in result.output and "not settled" in result.output

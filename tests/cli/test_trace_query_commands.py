"""``flowmesh trace tree`` / ``trace aggregate``, and the store-independence gate.

The CLI reaches the telemetry store only through the server, so the store stays
swappable without a client change. The import gate below is what keeps that true: a
convenience import of a store driver into the CLI package fails here.
"""

import ast
import pathlib
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import flowmesh
import flowmesh_cli
import pytest
import typer
from flowmesh.models import (
    TraceAggregate,
    TraceAggregateBucket,
    TraceSpanNode,
    TraceTree,
)
from flowmesh_cli.cli import build_cli_app
from typer.testing import CliRunner

runner = CliRunner()

_STORE_DRIVERS = ("clickhouse", "clickhouse_connect", "clickhouse_driver", "asynch")


def _app() -> typer.Typer:
    return build_cli_app()


def _node(
    span_id: str,
    name: str,
    *,
    seconds: float = 1.0,
    logical: dict[str, str] | None = None,
    physical: dict[str, str] | None = None,
    children: list[TraceSpanNode] | None = None,
) -> TraceSpanNode:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return TraceSpanNode(
        span_id=span_id,
        parent_span_id=None,
        name=name,
        start_time=start,
        end_time=start,
        duration_seconds=seconds,
        status="Unset",
        logical=logical or {},
        physical=physical or {},
        children=children or [],
    )


def _tree(roots: list[TraceSpanNode]) -> TraceTree:
    return TraceTree(
        workflow_id="wfl-1",
        trace_id="0" * 32,
        span_count=sum(1 + len(root.children) for root in roots),
        total_duration_seconds=2.5,
        roots=roots,
    )


def _client(traces: MagicMock) -> MagicMock:
    client = MagicMock()
    client.traces = traces
    return client


class TestTraceTree:
    def test_renders_children_indented_under_their_parent(self) -> None:
        traces = MagicMock()
        traces.tree.return_value = _tree(
            [
                _node(
                    "r",
                    "workflow",
                    seconds=2.5,
                    logical={"workflow_id": "wfl-1"},
                    children=[
                        _node(
                            "c",
                            "operator",
                            logical={"activation_id": "act-7"},
                            physical={"worker_id": "wkr-3"},
                        )
                    ],
                )
            ]
        )
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            result = runner.invoke(_app(), ["trace", "tree", "wfl-1"])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        parent = next(line for line in lines if "workflow" in line)
        child = next(line for line in lines if "operator" in line)
        assert not parent.startswith(" ")
        assert child.startswith("  ")
        assert "2.500s" in parent
        assert "activation_id=act-7" in child
        traces.tree.assert_called_once_with("wfl-1")

    def test_an_untraced_workflow_renders_without_an_error(self) -> None:
        traces = MagicMock()
        traces.tree.return_value = _tree([])
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            result = runner.invoke(_app(), ["trace", "tree", "wfl-1"])

        assert result.exit_code == 0
        assert "no spans recorded" in result.output

    def test_json_output_carries_both_attribute_views(self) -> None:
        traces = MagicMock()
        traces.tree.return_value = _tree(
            [_node("r", "workflow", logical={"scope_id": "scp-1"}, physical={"x": "y"})]
        )
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            result = runner.invoke(_app(), ["trace", "tree", "wfl-1", "--json"])

        assert result.exit_code == 0
        assert "logical" in result.output
        assert "physical" in result.output


class TestTraceAggregate:
    def test_passes_every_option_through_to_the_sdk(self) -> None:
        traces = MagicMock()
        traces.aggregate.return_value = TraceAggregate(
            metric="m",
            group_by="worker_id",
            stat="p95",
            buckets=[
                TraceAggregateBucket(
                    group_value="wkr-1", stat="p95", value=1.0, sample_count=3
                )
            ],
        )
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            result = runner.invoke(
                _app(),
                [
                    "trace",
                    "aggregate",
                    "--metric",
                    "m",
                    "--group-by",
                    "worker_id",
                    "--stat",
                    "p95",
                    "--kind",
                    "histogram",
                ],
            )

        assert result.exit_code == 0
        traces.aggregate.assert_called_once_with("m", "worker_id", "p95", "histogram")
        assert "wkr-1" in result.output

    def test_offers_no_workflow_scope_the_aggregate_cannot_honour(self) -> None:
        """No metric carries a workflow id, so the surface never advertises one."""
        traces = MagicMock()
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            help_text = runner.invoke(_app(), ["trace", "aggregate", "--help"]).output
            result = runner.invoke(
                _app(),
                [
                    "trace",
                    "aggregate",
                    "--metric",
                    "m",
                    "--group-by",
                    "worker_id",
                    "--workflow-id",
                    "wfl-1",
                ],
            )

        assert "--workflow-id" not in help_text
        assert result.exit_code != 0
        traces.aggregate.assert_not_called()

    def test_reports_an_empty_aggregate(self) -> None:
        traces = MagicMock()
        traces.aggregate.return_value = TraceAggregate(
            metric="m", group_by="worker_id", stat="avg"
        )
        with patch(
            "flowmesh_cli.commands.trace.FlowMesh", return_value=_client(traces)
        ):
            result = runner.invoke(
                _app(),
                ["trace", "aggregate", "--metric", "m", "--group-by", "worker_id"],
            )

        assert result.exit_code == 0
        assert "no datapoints" in result.output


def _imported_module_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", [flowmesh_cli, flowmesh], ids=["cli", "sdk"])
def test_no_client_package_imports_a_telemetry_store_driver(package) -> None:
    """A source scan, so the result cannot depend on what another test imported."""
    package_root = pathlib.Path(package.__path__[0])
    sources = sorted(package_root.rglob("*.py"))
    assert sources, f"{package.__name__} should contain modules to scan"

    offenders = {
        source.relative_to(package_root).as_posix(): sorted(hits)
        for source in sources
        if (
            hits := {
                name
                for name in _imported_module_names(source.read_text(encoding="utf-8"))
                if name.split(".")[0] in _STORE_DRIVERS
            }
        )
    }
    assert not offenders, (
        f"{package.__name__} reaches the telemetry store through the server, never a "
        f"store driver: {offenders}"
    )

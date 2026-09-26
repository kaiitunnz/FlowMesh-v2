"""A spawn region publishes its children's results as one keyed collection."""

from typing import Any

import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.validation import _check_result_declarations
from server.task.v2.representations.results import (
    CardinalityKind,
    ReleaseConditionKind,
    ResultDeclaration,
    Visibility,
)
from server.task.v2.representations.template import SourceMapEntry

_BINDINGS = AgentBindingDefaults(default_backend="codex")

_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: published}
spec:
  graph:
    nodes:
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: reviewer
        spec: {taskType: echo, data: {type: list, items: [tmpl]}}
      - name: fanout
        dependsOn: [planner]
        region: REGION
      - name: summarize
        dependsOn: [planner]
        spec:
          taskType: echo
          data: {type: list, items: [s]}
          v2: {result: {visibility: published}}
"""


def _compile(region: str) -> Any:
    text = _WF.replace("REGION", region)
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl-test", parsed, source, bindings=_BINDINGS)
    return template


def _codes(region: str) -> list[str]:
    with pytest.raises(CompileError) as caught:
        _compile(region)
    return [diag.code for diag in caught.value.diagnostics]


def test_a_published_spawn_declares_one_keyed_collection() -> None:
    template = _compile(
        "{kind: spawn, child: reviewer, result: {visibility: published}}"
    )

    published = dict(template.published_outputs())
    assert set(published) == {"fanout", "summarize"}
    decl = published["fanout"]
    assert decl.source_ref == "fanout"
    assert decl.cardinality is CardinalityKind.KEYED_COLLECTION
    assert decl.keying == "child_index"
    assert decl.release is ReleaseConditionKind.SOURCE_SETTLED
    assert decl.value_type == "echo"
    assert published["summarize"].cardinality is CardinalityKind.SINGLETON


def test_the_fixed_collection_fields_may_be_stated() -> None:
    template = _compile(
        "{kind: spawn, child: reviewer, result: {visibility: published, "
        "cardinality: keyed_collection, keying: child_index, release: source_settled}}"
    )
    assert "fanout" in dict(template.published_outputs())


def test_an_unpublished_spawn_declares_nothing() -> None:
    template = _compile("{kind: spawn, child: reviewer}")
    assert [name for name, _ in template.published_outputs()] == ["summarize"]
    assert all(d.source_ref != "fanout" for d in template.result_declarations)


@pytest.mark.parametrize(
    ("result", "code"),
    [
        ("{visibility: published, shape: list}", "region.result-unknown-field"),
        ("{visibility: internal}", "region.result-conflict"),
        ("{visibility: published, cardinality: singleton}", "region.result-conflict"),
        ("{visibility: published, keying: name}", "region.result-conflict"),
        ("{visibility: published, release: scope_closed}", "region.result-conflict"),
        ("published", "region.result-invalid"),
    ],
)
def test_a_malformed_result_is_a_targeted_error(result: str, code: str) -> None:
    assert _codes(f"{{kind: spawn, child: reviewer, result: {result}}}") == [code]


@pytest.mark.parametrize(
    "region",
    [
        "{kind: call, child: reviewer, result: {visibility: published}}",
        "{kind: join, completion: all_settled, result: {visibility: published}}",
        "{kind: merge, result: {visibility: published}}",
    ],
)
def test_only_a_spawn_publishes(region: str) -> None:
    assert _codes(region) == ["region.result-unsupported"]


def test_a_child_with_no_result_type_cannot_be_published() -> None:
    assert _codes("{kind: spawn, child: nothing, result: {visibility: published}}") == [
        "region.result-unresolved-child"
    ]


def test_a_call_never_publishes_its_single_child() -> None:
    template = _compile("{kind: call, child: reviewer}")
    assert all(
        decl.cardinality is not CardinalityKind.KEYED_COLLECTION
        for decl in template.result_declarations
    )
    assert [name for name, _ in template.published_outputs()] == ["summarize"]


def test_two_published_outputs_sharing_a_name_are_an_error() -> None:
    template = _compile(
        "{kind: spawn, child: reviewer, result: {visibility: published}}"
    )
    clashing = template.model_copy(
        update={
            "source_map": tuple(
                SourceMapEntry(
                    logical_ref=entry.logical_ref,
                    source_kind=entry.source_kind,
                    source_id=(
                        "fanout"
                        if entry.logical_ref == "summarize"
                        else entry.source_id
                    ),
                )
                for entry in template.source_map
            ),
            "result_declarations": template.result_declarations
            + (
                ResultDeclaration(
                    output_id="extra",
                    source_ref="summarize",
                    visibility=Visibility.PUBLISHED,
                ),
            ),
        }
    )
    codes = [d.code for d in _check_result_declarations(clashing, {})]
    assert "result.duplicate-name" in codes

"""An agent declares the workspace paths it exports: compilation pins them on its
operator and refuses credential paths and colliding names, its dispatch carries them,
and the completion that ends it fails when it could not copy them out."""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from server.task.models import TaskStatus
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.compiler.diagnostics import CompileError
from shared.harness import HarnessResult, HarnessResultKind
from shared.schemas.event import TaskEvent
from shared.tasks.components.exports import AgentExport, excluded_export_path
from shared.tasks.specs import AgentSpecStrict, AgentSpecTemplate
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_agent_episode_runtime import (
    _AGENT_WF,
    _HOLDER,
    _TS,
    _spawn_reviewer,
)
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_agent_binding_compile import (
    _agent,
    _agent_workflow,
    _compile,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime

_DEFAULTS = AgentBindingDefaults(default_backend="scripted")


@pytest.mark.parametrize("spec", [AgentSpecStrict, AgentSpecTemplate])
def test_both_spec_twins_carry_exports_named_by_their_path(spec: Any) -> None:
    parsed = spec.model_validate(
        {
            "taskType": "agent",
            "exports": [{"path": "out/report.md"}, {"path": "data", "name": "set"}],
        }
    )
    assert [e.published_name for e in parsed.exports] == ["out/report.md", "set"]


@pytest.mark.parametrize("path", ["../escape", "/abs", "a//b", "a/./b", ""])
def test_an_export_path_is_relative_and_plain(path: str) -> None:
    with pytest.raises(ValidationError):
        AgentExport(path=path)


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        (".ssh", ".ssh"),
        (".ssh/id_ed25519", ".ssh"),
        (".config/gcloud/creds.json", ".config/gcloud"),
        (".docker/config.json", ".docker/config.json"),
        (".config/gh/notes.md", None),
        ("ssh/keys", None),
    ],
)
def test_credential_paths_are_never_exported(path: str, excluded: str | None) -> None:
    assert excluded_export_path(path) == excluded


def test_compilation_pins_the_exports_on_the_agent() -> None:
    template, _ = _compile(
        _agent_workflow(
            """        exports: [{path: out/report.md}, {path: data, name: dataset}]"""
        ),
        _DEFAULTS,
    )
    assert _agent(template).exports == (
        AgentExport(path="out/report.md"),
        AgentExport(path="data", name="dataset"),
    )


@pytest.mark.parametrize(
    ("exports", "code"),
    [
        ("[{path: .aws/credentials}]", "agent.exports.excluded"),
        ("[{path: a}, {path: b, name: a}]", "agent.exports.name_conflict"),
        ("[{path: out}, {path: x, name: out/x}]", "agent.exports.name_conflict"),
    ],
)
def test_compilation_refuses_credential_paths_and_colliding_names(
    exports: str, code: str
) -> None:
    with pytest.raises(CompileError) as raised:
        _compile(_agent_workflow(f"        exports: {exports}"), _DEFAULTS)
    assert code in str(raised.value)


_WORKFLOW = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: exports}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          harness: {backend: scripted, version: v1, params: {script: []}}
          exports: [{path: out/report.md}]
"""


def _finished(kind: HarnessResultKind, failure: dict[str, str] | None) -> dict:
    step = (
        HarnessResult(kind=kind, value="done")
        if kind is HarnessResultKind.COMPLETION
        else HarnessResult(kind=kind)
    )
    payload: dict[str, Any] = {"agent_episode": step.model_dump(mode="json")}
    if failure is not None:
        payload["agent_export_failure"] = failure
    return payload


def _succeeded(task_id: str, payload: dict[str, Any]) -> TaskEvent:
    return TaskEvent(
        type="TASK_SUCCEEDED",
        task_id=task_id,
        worker_id="wkr-1",
        payload=payload,
        ts=_TS,
    )


_FAILURE = {"kind": "ExportUnsafe", "message": "export entry 'out/x' is a link"}


def test_the_dispatch_carries_the_exports_and_a_final_failure_fails_the_agent() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = asyncio.run(_register(runtime, _WORKFLOW))
    writer = ids["writer"]
    dispatch = runtime.agent_episode_dispatch(writer, _HOLDER)
    assert dispatch is not None
    assert dispatch.exports == (AgentExport(path="out/report.md"),)
    record_dispatch(runtime, writer, "wkr-1")

    _monitor(runtime).handle_task_event(
        _succeeded(writer, _finished(HarnessResultKind.COMPLETION, _FAILURE))
    )

    record = runtime.get_record(writer)
    assert record is not None
    assert record.status == TaskStatus.FAILED
    assert record.error is not None and record.error.startswith("ExportUnsafe:")


def test_a_completion_that_exports_cleanly_settles_the_agent() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = asyncio.run(_register(runtime, _WORKFLOW))
    writer = ids["writer"]
    record_dispatch(runtime, writer, "wkr-1")

    _monitor(runtime).handle_task_event(
        _succeeded(writer, _finished(HarnessResultKind.COMPLETION, None))
    )

    record = runtime.get_record(writer)
    assert record is not None and record.status == TaskStatus.DONE


def test_a_completion_that_yields_a_facade_group_ignores_its_export_failure() -> None:
    runtime = _runtime(FakeRegistry())
    workflow = _AGENT_WF.replace(
        "          harness: {backend: scripted, version: v1, params: {script: []}}\n",
        "          harness: {backend: scripted, version: v1, params: {script: []}}\n"
        "          exports: [{path: out/report.md}]\n",
    )
    _, ids = asyncio.run(_register(runtime, workflow))
    writer = ids["writer"]
    record_dispatch(runtime, writer, "wkr-1")
    runtime.originate_facade_turn_group(writer, _spawn_reviewer(writer))

    _monitor(runtime).handle_task_event(
        _succeeded(writer, _finished(HarnessResultKind.COMPLETION, _FAILURE))
    )

    record = runtime.get_record(writer)
    assert record is not None and record.status is TaskStatus.PENDING

"""SSH result mounting's parser and dispatch helper tests."""

import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.dispatcher.base import Dispatcher, StageReferenceNotReady
from server.registries.worker import WorkerRegistry
from server.task.models import TaskRecord, TaskStatus
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from shared.content import reference_for
from shared.schemas.result import ResultEnvelope
from shared.tasks import TaskEnvelopeTemplate, TaskType
from shared.tasks.result_binding import ResultBinding
from shared.tasks.specs import SSHSpecStrict


class _DummyRuntime:
    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        depends_on: dict[str, list[str]] | None = None,
        results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.tasks = tasks
        self._depends_on = depends_on or {}
        self._results = results or {}

    def read_result(self, task_id: str) -> ResultEnvelope | None:
        envelope = self._results.get(task_id)
        return ResultEnvelope.model_validate(envelope) if envelope else None

    def result_binding(self, task_id: str) -> ResultBinding | None:
        record = self.tasks.get(task_id)
        if record is None or record.status != TaskStatus.DONE:
            return None
        return ResultBinding(
            task_id=task_id,
            reference=reference_for(
                "org", task_id.encode(), media_type="application/json"
            ),
        )

    def get_record(self, task_id: str) -> TaskRecord | None:
        return self.tasks.get(task_id)

    def describe_task(self, task_id: str) -> SimpleNamespace | None:
        record = self.tasks.get(task_id)
        if record is None:
            return None
        return SimpleNamespace(depends_on=list(self._depends_on.get(task_id, [])))


def _task_template(task_type: TaskType, **spec_updates: object) -> TaskEnvelopeTemplate:
    payload = {
        "apiVersion": "flowmesh/v1",
        "kind": "Task",
        "metadata": {"name": "wf:task"},
        "spec": {"taskType": task_type.value, **spec_updates},
    }
    return TaskEnvelopeTemplate.model_validate(payload)


def test_parse_workflow_preserves_stage_local_names_for_ssh_inputs() -> None:
    payload = textwrap.dedent("""
        apiVersion: flowmesh/v1
        kind: Workflow
        metadata:
          name: wf
        spec:
          stages:
            - name: preprocess
              spec:
                taskType: echo
            - name: annotate
              dependsOn: [preprocess]
              spec:
                taskType: ssh
                inputs:
                  - stage: preprocess
        """)

    parsed = parse_workflow(payload, "native")

    assert [task.local_name for task in parsed.tasks] == ["preprocess", "annotate"]


def test_dispatcher_resolves_ssh_input_stage_names_from_local_stage_names() -> None:
    upstream = TaskRecord(
        task_id="task-pre",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="preprocess",
    )
    current_task = _task_template(
        TaskType.SSH,
        inputs=[{"stage": "preprocess"}],
        accessMode="direct",
    )
    current = TaskRecord(
        task_id="task-ssh",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=current_task,
        status=TaskStatus.PENDING,
        task_type="ssh",
        local_name="annotate",
    )

    dispatcher = Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {upstream.task_id: upstream, current.task_id: current},
                depends_on={current.task_id: [upstream.task_id]},
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("test-ssh-phase2"),
    )

    spec = SSHSpecStrict.model_validate(current.task.spec.model_dump())
    dispatcher._validate_ssh_inputs(current, spec)  # noqa: SLF001
    _, upstream_results = dispatcher._resolve_stage_references(  # noqa: SLF001
        current.task_id, current.task, current
    )

    assert upstream_results is not None
    assert upstream_results["preprocess"].task_id == "task-pre"
    assert upstream_results["preprocess"].reference is not None


def test_dispatcher_requeues_when_ssh_input_stage_not_done() -> None:
    upstream = TaskRecord(
        task_id="task-pre",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.PENDING,
        task_type="echo",
        local_name="preprocess",
    )
    current = TaskRecord(
        task_id="task-ssh",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.SSH, inputs=[{"stage": "preprocess"}]),
        status=TaskStatus.PENDING,
        task_type="ssh",
        local_name="annotate",
    )
    dispatcher = Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {upstream.task_id: upstream, current.task_id: current},
                depends_on={current.task_id: [upstream.task_id]},
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("test-ssh-phase2"),
    )
    spec = SSHSpecStrict.model_validate(current.task.spec.model_dump())

    with pytest.raises(StageReferenceNotReady):
        dispatcher._validate_ssh_inputs(current, spec)  # noqa: SLF001


def test_build_stage_context_includes_only_transitive_dependencies() -> None:
    upstream = TaskRecord(
        task_id="task-pre",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="preprocess",
    )
    middle = TaskRecord(
        task_id="task-mid",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="transform",
    )
    unrelated = TaskRecord(
        task_id="task-unrelated",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="other-branch",
    )
    current = TaskRecord(
        task_id="task-final",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(
            TaskType.ECHO,
            data={"message": "${preprocess.responses.0.output}"},
        ),
        status=TaskStatus.PENDING,
        task_type="echo",
        local_name="finalize",
    )
    dispatcher = Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {
                    upstream.task_id: upstream,
                    middle.task_id: middle,
                    unrelated.task_id: unrelated,
                    current.task_id: current,
                },
                depends_on={
                    current.task_id: [middle.task_id],
                    middle.task_id: [upstream.task_id],
                },
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("test-stage-context"),
    )

    context = dispatcher._build_stage_context(current)  # noqa: SLF001

    assert set(context) == {"preprocess", "transform"}


def test_upstream_bindings_exclude_unrelated_completed_stages(
    tmp_path: Path,
) -> None:
    results = {
        "task-pre": {
            "task_id": "task-pre",
            "result": {"responses": [{"output": "pre"}]},
        },
        "task-other": {
            "task_id": "task-other",
            "result": {"responses": [{"output": "other"}]},
        },
    }

    upstream = TaskRecord(
        task_id="task-pre",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="preprocess",
    )
    unrelated = TaskRecord(
        task_id="task-other",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="other-branch",
    )
    current = TaskRecord(
        task_id="task-final",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(
            TaskType.ECHO,
            data={"message": "${preprocess.responses.0.output}"},
        ),
        status=TaskStatus.PENDING,
        task_type="echo",
        local_name="finalize",
    )

    dispatcher = Dispatcher(
        runtime=cast(
            TaskRuntime,
            _DummyRuntime(
                {
                    upstream.task_id: upstream,
                    unrelated.task_id: unrelated,
                    current.task_id: current,
                },
                depends_on={current.task_id: [upstream.task_id]},
                results=results,
            ),
        ),
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("test-stage-results"),
    )

    context = dispatcher._build_stage_context(current)  # noqa: SLF001
    upstream_results = dispatcher._upstream_bindings(  # noqa: SLF001
        context, current.task_id
    )

    assert set(upstream_results) == {"preprocess"}


def test_stage_reference_uses_payload_root_for_local_and_http_results(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "task-local"
    http_dir = tmp_path / "task-http"
    results: dict[str, dict[str, Any]] = {
        "task-local": {
            "task_id": "task-local",
            "result": {
                "final_lora_archive": {"path": "final_lora.tar.gz"},
                "_artifacts": {"base_dir": local_dir.as_posix(), "base_url": None},
            },
        },
        "task-http": {
            "task_id": "task-http",
            "worker_id": "worker-1",
            "metadata": None,
            "received_at": "2026-05-10T00:00:00+00:00",
            "result": {
                "final_lora_archive": {"path": "final_lora.tar.gz"},
                "_artifacts": {
                    "base_dir": http_dir.as_posix(),
                    "base_url": "http://flowmesh.example",
                },
            },
        },
    }

    local_record = TaskRecord(
        task_id="task-local",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="local",
    )
    http_record = TaskRecord(
        task_id="task-http",
        workflow_id="wf-1",
        owner_id="owner",
        raw_yaml="raw",
        task=_task_template(TaskType.ECHO),
        status=TaskStatus.DONE,
        task_type="echo",
        local_name="http",
    )
    dispatcher = Dispatcher(
        runtime=cast(TaskRuntime, _DummyRuntime({}, results=results)),
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("test-stage-reference-root"),
    )

    local_value = dispatcher._resolve_reference(  # noqa: SLF001
        "local.final_lora_archive", {"local": local_record}
    )
    http_value = dispatcher._resolve_reference(  # noqa: SLF001
        "http.final_lora_archive", {"http": http_record}
    )

    assert local_value == (local_dir / "artifacts" / "final_lora.tar.gz").as_posix()
    assert (
        http_value
        == "http://flowmesh.example/api/v1/results/task-http/files/final_lora.tar.gz"
    )

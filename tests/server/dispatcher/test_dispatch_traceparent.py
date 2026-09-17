"""The task dispatch carrier: ``WorkerTaskMessage.traceparent``.

The dispatcher never derives a traceparent itself: it reads ``TaskRuntime``'s read-only
accessor and forwards the result onto the published ``WorkerTaskMessage``, so a worker's
run can parent on its episode span.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.tasks.worker_message import WorkerTaskMessage
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault
from tests.server.telemetry_helpers import recording_control_tracer

_ECHO_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: tp-dispatch
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""

_ECHO_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: tp-dispatch-v2}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""

_TP = "00-11111111111111111111111111111111-2222222222222222-01"


def _decode_traceparent(traceparent: str) -> tuple[str, str]:
    _version, trace_id_hex, span_id_hex, _flags = traceparent.split("-")
    return trace_id_hex, span_id_hex


def _runtime(control: Any = None) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("dispatch-traceparent-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        control=control,
    )


def _register(runtime: TaskRuntime) -> str:
    _workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _ECHO_WORKFLOW, format="native")
    )
    return results[0].task_id


def _dispatch(runtime: TaskRuntime, task_id: str, control: Any = None) -> mock.Mock:
    worker = Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    disp = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("dispatch-traceparent-dispatch"),
        control=control,
    )
    disp.dispatch_once(task_id)
    return registry


def test_a_dispatched_task_carries_a_stubbed_accessors_traceparent() -> None:
    # Exercises the dispatcher's own wiring -- it forwards whatever the accessor
    # returns -- independent of the accessor's real derivation, which the next test
    # covers end to end.
    control, _exporter = recording_control_tracer()
    runtime = _runtime(control=control)
    task_id = _register(runtime)
    cast(Any, runtime).dispatch_traceparent = lambda _task_id: _TP

    registry = _dispatch(runtime, task_id, control=control)

    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    assert message.traceparent == _TP


def test_a_v2_dispatch_names_the_workflows_trace_and_the_episodes_span() -> None:
    # End to end through the real accessor: the value the dispatcher forwards is the
    # workflow's derived trace id and the dispatched task's episode (work item) span id
    # -- not the attempt, which does not exist yet at publish time.
    control, _exporter = recording_control_tracer()
    runtime = _runtime(control=control)
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _ECHO_V2, format="native")
    )
    task_id = results[0].task_id
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item_id = engine.work_item_id_for_task(task_id)
    assert work_item_id is not None

    registry = _dispatch(runtime, task_id, control=control)

    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    assert message.traceparent is not None
    trace_id_hex, span_id_hex = _decode_traceparent(message.traceparent)
    assert trace_id_hex == f"{workflow_to_trace_id_int(workflow_id):032x}"
    assert span_id_hex == f"{derived_span_id(SpanIdKind.WORK_ITEM, work_item_id):016x}"


def test_disabled_telemetry_emits_no_traceparent_field() -> None:
    runtime = _runtime(control=None)
    task_id = _register(runtime)

    registry = _dispatch(runtime, task_id, control=None)

    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    assert message.traceparent is None
    wire = message.model_dump(mode="json", exclude_none=True, by_alias=True)
    assert "traceparent" not in wire


def test_a_v1_task_dispatches_with_no_traceparent() -> None:
    # dispatch_traceparent names no work item for a v1 task, so the field is absent
    # even with telemetry on.
    control, _exporter = recording_control_tracer()
    runtime = _runtime(control=control)
    task_id = _register(runtime)

    registry = _dispatch(runtime, task_id, control=control)

    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    assert message.traceparent is None

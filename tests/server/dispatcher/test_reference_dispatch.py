"""A dispatch names the upstream results a task consumes; it never carries them.

The task's worker hydrates each named result itself, so the message control publishes —
and the record control persists — hold references, and control reads no stored result
for a task that renders no placeholder from one.
"""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault

_PAYLOAD = "a-payload-only-its-consumer-may-read"

_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [alpha]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [beta]}}
      - name: c
        dependsOn: [b]
        spec: {taskType: echo, data: {type: list, items: [gamma]}}
"""

_SSH = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: ssh-input}
spec:
  stages:
    - name: preprocess
      spec: {taskType: echo, data: {type: list, items: [alpha]}}
    - name: annotate
      dependsOn: [preprocess]
      spec:
        taskType: ssh
        accessMode: direct
        inputs: [{stage: preprocess}]
"""


def _worker() -> Worker:
    return Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="n",
        node_alias="n",
        incarnation=1,
    )


class _Workflow:
    def __init__(self, payload: str) -> None:
        self.reader = make_result_reader()
        self.runtime = TaskRuntime(
            cast(Any, FakeRegistry()),
            cast(Any, mock.Mock()),
            OrchestrationConfig(),
            self.reader,
            logging.getLogger("reference-dispatch"),
            secret_vault=cast(Any, _NoopSecretVault()),
        )
        _wf, results = asyncio.run(
            self.runtime.register("owner", "org", payload, format="native")
        )
        self.ids = [r.task_id for r in results]
        self.registry = mock.Mock()
        self.registry.idle_satisfying_pool.return_value = [_worker()]
        self.registry.satisfying_workers.return_value = [_worker()]
        self.registry.publish_task.return_value = 1
        self.registry.get_worker.return_value = _worker()
        self.dispatcher = CapturingDispatcher(
            runtime=self.runtime,
            worker_registry=self.registry,
            logger=logging.getLogger("dispatch"),
        )

    def settle(self, task_id: str) -> None:
        record_dispatch(self.runtime, task_id, cast(Any, _worker()))
        self.runtime.mark_succeeded(
            task_id,
            "wkr-1",
            result_payload(self.reader, task_id, {"items": [_PAYLOAD]}, "org"),
            "t",
        )

    def dispatch(self, task_id: str) -> WorkerTaskMessage:
        reads = mock.Mock(wraps=self.reader._store.fetch)
        self.reader._store.fetch = reads  # type: ignore[method-assign]
        self.reader._cache.clear()
        assert self.dispatcher.dispatch_once(task_id)
        assert reads.call_count == 0, "control read a result it only forwards"
        message = self.registry.publish_task.call_args.args[1]
        assert isinstance(message, WorkerTaskMessage)
        return message


def test_a_dispatch_names_every_transitive_upstream_and_carries_none() -> None:
    wf = _Workflow(_CHAIN)
    a, b, c = wf.ids
    wf.settle(a)
    wf.settle(b)

    message = wf.dispatch(c)

    assert message.upstream_results is not None
    assert set(message.upstream_results) == {"a", "b"}
    for stage, task_id in (("a", a), ("b", b)):
        binding = wf.runtime.result_binding(task_id)
        assert binding is not None
        assert message.upstream_results[stage] == binding
    assert message.spec.upstreamResults is None
    wire = message.model_dump_json(exclude_none=True, by_alias=True)
    assert _PAYLOAD not in wire
    assert _PAYLOAD not in wf.runtime._tasks[c].model_dump_json()


def test_an_ssh_input_is_named_by_its_upstream_binding() -> None:
    wf = _Workflow(_SSH)
    preprocess, annotate = wf.ids
    wf.settle(preprocess)

    message = wf.dispatch(annotate)

    assert message.upstream_task_ids is None
    assert message.upstream_results is not None
    assert message.upstream_results["preprocess"].task_id == preprocess
    assert _PAYLOAD not in message.model_dump_json(exclude_none=True, by_alias=True)

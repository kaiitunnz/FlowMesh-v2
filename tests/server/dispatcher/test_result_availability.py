"""A dispatch that reads a stored result waits out an unreachable store.

A downstream task's stage reference is resolved from its producer's stored result.
When the store cannot be reached the task is requeued without spending an attempt,
since the result is still there; when the result is missing or corrupt it fails at
once rather than retrying toward a result that will never read.
"""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.content import (
    OCTET_STREAM,
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
)
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault

_DAG = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: stage-reference}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [alpha]}}
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
          data:
            type: list
            items: ["${a.items.0.output}"]
"""


class _FlakyStore(FabricObjectStore):
    """A store whose reads fail with a chosen error while one is set."""

    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store
        self.error: Exception | None = None

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._store.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        if self.error is not None:
            raise self.error
        return self._store.fetch(reference)


def _worker() -> Worker:
    return Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="n",
        node_alias="n",
        incarnation=1,
    )


def _dispatch_downstream(error: Exception) -> CapturingDispatcher:
    reader = make_result_reader()
    runtime = TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("result-availability"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )
    _wf, results = asyncio.run(runtime.register("owner", "org", _DAG, format="native"))
    a, b = (r.task_id for r in results)
    runtime.mark_dispatched(a, cast(Any, _worker()))
    runtime.mark_succeeded(
        a,
        "wkr-1",
        result_payload(reader, a, {"items": [{"output": "alpha"}]}, "org"),
        "t",
    )
    flaky = _FlakyStore(reader._store)
    reader._store = flaky
    flaky.error = error
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker()]
    registry.satisfying_workers.return_value = [_worker()]
    registry.publish_task.return_value = 1
    registry.get_worker.return_value = _worker()
    dispatcher = CapturingDispatcher(
        runtime=runtime, worker_registry=registry, logger=logging.getLogger("dispatch")
    )
    dispatcher.dispatch_once(b)
    return dispatcher


def test_an_unreachable_store_requeues_without_spending_an_attempt() -> None:
    dispatcher = _dispatch_downstream(ContentUnavailable("store down"))
    assert dispatcher.failed == []
    ((_task, kwargs),) = dispatcher.requeued
    assert kwargs["count_retry"] is False


def test_a_missing_or_corrupt_result_fails_the_task_at_once() -> None:
    dispatcher = _dispatch_downstream(ContentHydrationError("no such object"))
    assert dispatcher.requeued == []
    assert len(dispatcher.failed) == 1

"""A dispatch that reads a stored result waits out an unreachable store.

A downstream task's stage reference is resolved from its producer's stored result.
When the store cannot be reached the task is requeued without spending an attempt,
since the result is still there; when the result is missing or corrupt it fails at
once rather than retrying toward a result that will never read.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.results import ResultUnavailable
from server.task.runtime import TaskRuntime
from shared.content import (
    OCTET_STREAM,
    TREE_MEDIA_TYPE,
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
    TreeManifest,
    TreeMember,
)
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry

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
# b loads a's published model, which its reference renders as a token.
_NAMED_DAG = _DAG.replace("${a.items.0.output}", "load ${a.items.0.model}")


class _FlakyStore(FabricObjectStore):
    """A store whose reads fail with a chosen error while one is set."""

    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store
        self.error: Exception | None = None
        self.only: str | None = None

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        return self._store.write(scope, data, media_type=media_type)

    def fetch(self, reference: ContentReference) -> bytes:
        if self.error is not None and self.only in (None, reference.content_digest):
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


def _dispatch_downstream(
    error: Exception, *, tree_only: bool = False, named: bool = False
) -> CapturingDispatcher:
    reader = make_result_reader()
    runtime = TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("result-availability"),
        credential_vault=InMemoryCredentialVault(),
    )
    _wf, results = asyncio.run(
        runtime.register("owner", "org", _NAMED_DAG if named else _DAG, format="native")
    )
    a, b = (r.task_id for r in results)
    record_dispatch(runtime, a, cast(Any, _worker()))
    member = reader._store.write("org", b"weights")
    manifest = TreeManifest.of(
        [TreeMember("w.bin", member.content_digest, member.size_bytes)]
    )
    tree = reader._store.write("org", manifest.encode(), media_type=TREE_MEDIA_TYPE)
    payload = result_payload(
        reader,
        a,
        {
            "items": [{"output": "alpha", "model": {"path": "final_model"}}],
            "_artifacts": {"task_id": a},
        },
        "org",
    )
    payload["artifacts"] = [
        ArtifactBinding(
            name="final_model", kind=ArtifactKind.TREE, reference=tree, size_bytes=7
        ).model_dump(mode="json")
    ]
    runtime.mark_succeeded(a, "wkr-1", payload, "t")
    flaky = _FlakyStore(reader._store)
    reader._store = flaky
    flaky.error = error
    if tree_only:
        flaky.only = tree.content_digest
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


def test_an_unreachable_tree_the_task_loads_holds_it() -> None:
    dispatcher = _dispatch_downstream(
        ContentUnavailable("store down"), tree_only=True, named=True
    )
    assert dispatcher.failed == []
    ((_task, kwargs),) = dispatcher.requeued
    assert kwargs == {"reason": "artifact_input_unavailable", "count_retry": False}


def test_a_missing_tree_the_task_loads_fails_it() -> None:
    dispatcher = _dispatch_downstream(
        ContentHydrationError("no such object"), tree_only=True, named=True
    )
    assert dispatcher.requeued == []
    assert len(dispatcher.failed) == 1


@pytest.mark.parametrize(
    "error", [ContentUnavailable("store down"), ContentHydrationError("no such object")]
)
def test_a_tree_only_an_upstream_value_reaches_never_holds_or_fails_the_task(
    error: Exception,
) -> None:
    dispatcher = _dispatch_downstream(error, tree_only=True)
    assert dispatcher.requeued == []
    assert dispatcher.failed == []
    cast(Any, dispatcher._worker_registry).publish_task.assert_called_once()


def test_an_unreachable_store_ends_the_reading_of_lent_trees() -> None:
    reads: list[str] = []

    class _Down:
        def tree_members(self, reference: ContentReference) -> Any:
            reads.append(reference.content_digest)
            raise ResultUnavailable("store down")

    lent = [
        ArtifactBinding(
            name="final_model",
            kind=ArtifactKind.TREE,
            reference=ContentReference(
                authorization_scope="org",
                content_digest=f"{n:064x}",
                size_bytes=1,
            ),
            size_bytes=1,
        )
        for n in range(3)
    ]
    runtime = cast(
        Any, SimpleNamespace(_results=_Down(), _logger=logging.getLogger("t"))
    )

    (objects,) = TaskRuntime.artifact_input_objects(runtime, [], lent)

    assert objects == {binding.reference.identity for binding in lent}
    assert len(reads) == 1

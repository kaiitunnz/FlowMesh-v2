"""What a dispatched task is given to reach the shared content store with.

The scope a task writes under is control's to assign, and the access it opens the store
with is minted per dispatch and relayed over the worker's own attachment. Both are the
dispatcher's doing, so both are asserted where the dispatch happens.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.content import ContentAccessBroker
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.content import (
    ContentOperationKind,
    ContentStoreAccess,
    ScopedContentCredential,
)
from shared.tasks.worker_message import WorkerTaskMessage
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault

_ORG = "org-acme"

_ECHO_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: content-scope
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""


class _RecordingMinter:
    """Cuts a stand-in credential, remembering what it was asked for."""

    def __init__(self, fails: bool = False) -> None:
        self.asked: list[tuple[str, tuple[ContentOperationKind, ...], float]] = []
        self._fails = fails

    @property
    def policy_version(self) -> str:
        return "test-1"

    def mint(
        self, scope: str, operations: tuple[ContentOperationKind, ...], ttl_sec: float
    ) -> ScopedContentCredential:
        self.asked.append((scope, operations, ttl_sec))
        if self._fails:
            raise RuntimeError("the backend cut no session")
        return ScopedContentCredential(material={"token": "opaque"})


def _worker() -> Worker:
    return Worker(
        id="wkr-1",
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=7,
    )


def _runtime() -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("content-scope-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _dispatch(minter: _RecordingMinter) -> mock.Mock:
    runtime = _runtime()
    _workflow_id, results = asyncio.run(
        runtime.register("owner", _ORG, _ECHO_WORKFLOW, format="native")
    )
    worker = _worker()
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [worker]
    registry.satisfying_workers.return_value = [worker]
    registry.publish_task.return_value = 1
    registry.get_worker.return_value = worker
    broker = ContentAccessBroker(registry, minter, grant_ttl_sec=900.0)
    CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("content-scope-dispatch"),
        content_access=broker,
    ).dispatch_once(results[0].task_id)
    return registry


def _relayed_access(registry: mock.Mock) -> ContentStoreAccess | None:
    for call in registry.publish_mediated_op.call_args_list:
        message = call[0][1]
        if message.frame_kind == "content_access":
            return ContentStoreAccess.model_validate(message.payload)
    return None


def test_a_dispatched_task_carries_the_scope_its_owner_writes_under() -> None:
    registry = _dispatch(_RecordingMinter())
    message = registry.publish_task.call_args[0][1]
    assert isinstance(message, WorkerTaskMessage)
    assert message.content_scope == _ORG


def test_the_task_is_given_access_for_exactly_that_scope() -> None:
    minter = _RecordingMinter()
    registry = _dispatch(minter)

    assert [scope for scope, _ops, _ttl in minter.asked] == [_ORG]
    access = _relayed_access(registry)
    assert access is not None
    assert access.grant.authorization_scope == _ORG
    assert access.grant.subject == "wkr-1" and access.grant.subject_generation == 7
    assert access.credential.material == {"token": "opaque"}


def test_the_access_is_relayed_before_the_task_is_published() -> None:
    registry = _dispatch(_RecordingMinter())
    order = [
        call[0]
        for call in registry.method_calls
        if call[0] in {"publish_mediated_op", "publish_task"}
    ]
    assert order.index("publish_mediated_op") < order.index("publish_task")


def test_a_backend_that_cuts_no_session_relays_no_access() -> None:
    registry = _dispatch(_RecordingMinter(fails=True))
    assert _relayed_access(registry) is None
    assert registry.publish_task.called

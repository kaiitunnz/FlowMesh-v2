"""A task prefers an idle worker already holding a copy of what its upstream tasks
published, and never waits for one."""

import asyncio
import hashlib
import logging
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from server.content.directory import ContentHolderRecord
from server.registries.artifacts import final_set
from server.registries.worker import Worker
from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    make_capturing_dispatcher,
)
from tests.server.result_store import result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.server.task.test_v2_orchestration import _runtime as _real_runtime
from tests.support.waiting import pop_ready

_TS = "2026-10-14T00:00:00Z"


def _ref(data: bytes, size: int) -> ContentReference:
    return ContentReference(
        authorization_scope="org",
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=size,
    )


_WEIGHTS = _ref(b"weights", 5_000)
_TOKENIZER = _ref(b"tokenizer", 10)


def _tree(name: str, largest: ContentReference, revision: str = "final") -> Any:
    return ArtifactBinding(
        name=name,
        revision=revision,
        kind=ArtifactKind.TREE,
        reference=_ref(name.encode(), 20),
        size_bytes=largest.size_bytes,
        largest=largest,
    )


class _Runtime:
    def __init__(self, scoped: dict[str, Any] | None, depends_on: list[str]) -> None:
        self._scoped = scoped
        self._depends_on = depends_on
        self.artifacts = {
            "tsk-train": {
                ("final", "model"): _tree("model", _WEIGHTS),
                ("final", "tok"): _tree("tok", _TOKENIZER),
                ("step-2", "checkpoints"): _tree(
                    "checkpoints", _ref(b"c", 9_000), "step-2"
                ),
            }
        }

    def scoped_inputs(self, task_id: str) -> dict[str, Any] | None:
        return self._scoped

    def describe_task(self, task_id: str) -> Any:
        return SimpleNamespace(depends_on=self._depends_on)

    def final_artifacts(self, task_id: str) -> Any:
        return final_set(self.artifacts.get(task_id) or {})


class _Holders:
    def __init__(self, by_digest: dict[str, list[ContentHolderRecord]]) -> None:
        self._by_digest = by_digest
        self.reads: list[str] = []

    def holders(self, scope: str, digest: str) -> list[ContentHolderRecord]:
        self.reads.append(digest)
        return self._by_digest.get(digest, [])


def _holder(worker_id: str, generation: int = 1) -> ContentHolderRecord:
    return ContentHolderRecord(worker_id, "nde-1", generation, 2e9)


def _pool(*ids: str) -> list[Any]:
    return [SimpleNamespace(id=worker_id, incarnation=1) for worker_id in ids]


def _dispatcher(holders: _Holders | None, v2: bool = False) -> Any:
    scoped = (
        {"model": SimpleNamespace(task_id="tsk-train", binding=None)} if v2 else None
    )
    return make_capturing_dispatcher(
        runtime=_Runtime(scoped, [] if v2 else ["tsk-train"]),
        content_holders=holders,
    )


@pytest.mark.parametrize("v2", [False, True])
def test_an_idle_holder_of_the_largest_upstream_object_is_preferred(v2: bool) -> None:
    holders = _Holders(
        {
            _WEIGHTS.content_digest: [_holder("wkr-2")],
            _TOKENIZER.content_digest: [_holder("wkr-3")],
        }
    )
    chosen = _dispatcher(holders, v2)._preferred_holder(
        "tsk-1", _pool("wkr-1", "wkr-2", "wkr-3")
    )

    assert chosen is not None and chosen.id == "wkr-2"
    assert holders.reads == [_WEIGHTS.content_digest]


def test_a_holder_of_another_incarnation_is_not_preferred() -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-2", generation=7)]})

    assert _dispatcher(holders)._preferred_holder("tsk-1", _pool("wkr-2")) is None


def test_a_busy_holder_leaves_the_task_to_normal_selection() -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-busy")]})
    dispatcher = _dispatcher(holders)

    assert dispatcher._preferred_holder("tsk-1", _pool("wkr-1")) is None
    assert dispatcher.requeued == []


def test_without_a_holder_directory_nothing_is_read() -> None:
    assert _dispatcher(None)._preferred_holder("tsk-1", _pool("wkr-1")) is None


_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - {name: train, spec: {taskType: echo}}
      - {name: use, dependsOn: [train], spec: {taskType: echo}}
"""


def _worker(worker_id: str) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="c",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )


def _dispatch_use(enabled: bool, holders: Any) -> tuple[Any, mock.Mock, dict, Any]:
    runtime = _real_runtime(FakeRegistry())
    _, ids = asyncio.run(_register(runtime, _CHAIN))
    train = ids["train"]
    assert pop_ready(runtime) == train
    record_dispatch(runtime, train, "wkr-1", "dsp-1")
    runtime.mark_started(train, "wkr-1", {}, _TS)
    payload = result_payload(runtime._results, train, {"ok": True}, "org")
    model = _tree("model", _WEIGHTS)
    payload["artifacts"] = [model.model_dump(mode="json")]
    runtime.mark_succeeded(train, "wkr-1", payload, _TS, "dsp-1")
    use = ids["use"]
    assert pop_ready(runtime) == use
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker("wkr-1"), _worker("wkr-2")]
    registry.satisfying_workers.return_value = [_worker("wkr-1"), _worker("wkr-2")]
    registry.publish_task.return_value = 1
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("holder-preference"),
        enable_stage_weight_stickiness=enabled,
        content_holders=holders,
        worker_selection_strategy="first_fit",
    )
    dispatcher.dispatch_once(use)
    return runtime, registry, ids, model


@pytest.mark.parametrize("enabled", [True, False])
def test_a_dispatch_goes_to_the_holder_only_with_stickiness_on(enabled: bool) -> None:
    holders = _Holders({_WEIGHTS.content_digest: [_holder("wkr-2")]})

    _, registry, _, _ = _dispatch_use(enabled, holders)

    assert registry.publish_task.call_args[0][0].id == ("wkr-2" if enabled else "wkr-1")
    assert holders.reads == ([_WEIGHTS.content_digest] if enabled else [])


def test_a_dispatch_carries_its_upstream_artifacts_and_its_worker_may_read_them() -> (
    None
):
    runtime, registry, ids, model = _dispatch_use(False, None)

    worker, message = registry.publish_task.call_args[0]
    assert message.upstream_artifacts is not None
    assert [b.reference for b in message.upstream_artifacts[ids["train"]]] == [
        model.reference
    ]
    assert runtime.content_binding_authorizes(ids["use"], worker.id, model.reference)
    assert not runtime.content_binding_authorizes(
        ids["use"], worker.id, _ref(b"elsewhere", 3)
    )

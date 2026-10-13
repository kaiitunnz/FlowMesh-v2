"""A task's success binds the artifacts it published, once, in the transaction that
settles it, and the catalog survives a restart."""

import asyncio
import hashlib
import logging
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import task_artifacts_key
from server.config import OrchestrationConfig
from server.registries.workflow import WorkflowRegistry
from server.task.models import EventEffect, TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime, TransitionNotDurable
from shared.content import ContentReference
from shared.tasks.artifact_binding import ArchiveFacts, ArtifactBinding, ArtifactKind
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import _TS, _WorkerRegistryStub

_V1 = """apiVersion: flowmesh/v1
kind: EchoTask
metadata: {name: artifacts}
spec:
  taskType: echo
  stages:
    - name: train
      spec:
        data: {type: list, items: [a]}
    - name: use
      spec:
        data: {type: list, items: [b]}
"""


class _FlakyRegistry(WorkflowRegistry):
    """Refuses the next ``refuse`` record commits before applying them."""

    refuse = 0

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        if self.refuse:
            self.refuse -= 1
            raise ConnectionError("control store refused the commit")
        super().commit_transition(workflow_id, **kwargs)


def _registry() -> _FlakyRegistry:
    return _FlakyRegistry(fake_redis_client(fakeredis.FakeServer()))


def _runtime(registry: WorkflowRegistry, reader: Any = None) -> TaskRuntime:
    return TaskRuntime(
        registry,
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader or make_result_reader(),
        logging.getLogger("test.artifact_catalog"),
        credential_vault=InMemoryCredentialVault(),
        redrive=lambda fire, logger: StoreRedriveScheduler(
            fire, logger, run_thread=False
        ),
    )


def _register(runtime: TaskRuntime) -> tuple[str, list[str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _V1, format="native")
    )
    return workflow_id, [result.task_id for result in results]


def _ref(data: bytes, scope: str = "org") -> ContentReference:
    return ContentReference(
        authorization_scope=scope,
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _tree(name: str = "final_model", data: bytes = b"tree") -> ArtifactBinding:
    return ArtifactBinding(
        name=name,
        kind=ArtifactKind.TREE,
        reference=_ref(data),
        size_bytes=10,
        files=2,
        largest=_ref(b"largest"),
    )


def _archive(of: str = "final_model", data: bytes = b"tree") -> ArtifactBinding:
    return ArtifactBinding(
        name=f"{of}.tar.gz",
        kind=ArtifactKind.ARCHIVE,
        reference=_ref(data),
        size_bytes=30,
        of=of,
        archive=ArchiveFacts(sha256="ab" * 32, crc32=7, size_bytes=30, top=of),
    )


def _succeed(
    runtime: TaskRuntime,
    task_id: str,
    artifacts: list[ArtifactBinding] | None,
    dispatch_id: str | None = None,
) -> Any:
    payload = result_payload(runtime._results, task_id, {"ok": True}, "org")
    if artifacts is not None:
        payload["artifacts"] = [a.model_dump(mode="json") for a in artifacts]
    return runtime.mark_succeeded(
        task_id, "wkr-1", payload, _TS, dispatch_id=dispatch_id
    )


def _stored(registry: WorkflowRegistry, task_id: str) -> dict[tuple[str, str], Any]:
    return registry.artifact_bindings(task_id)


def test_a_success_binds_its_artifacts_with_its_record() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")

    _succeed(runtime, train, [_tree(), _archive()], "dsp-1")

    stored = _stored(registry, train)
    assert set(stored) == {("final", "final_model"), ("final", "final_model.tar.gz")}
    assert {b.dispatch_id for b in stored.values()} == {"dsp-1"}
    assert all(b.published_at for b in stored.values())
    assert runtime._tasks[train].artifacts == stored
    assert runtime._tasks[train].unwritten_artifacts == []


def test_a_replayed_success_adds_nothing() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    _succeed(runtime, train, [_tree()], "dsp-1")
    first = _stored(registry, train)

    replay = _succeed(runtime, train, [_tree(data=b"other"), _tree("extra")], "dsp-1")

    assert replay.effect is EventEffect.SETTLED
    assert _stored(registry, train) == first


def test_a_stale_dispatch_binds_nothing() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")

    outcome = _succeed(runtime, train, [_tree()], "dsp-0")

    assert outcome.effect is EventEffect.STALE
    assert raw_control(registry).exists(task_artifacts_key(train)) == 0


def test_artifacts_outside_the_scope_or_over_no_tree_are_refused() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    foreign = _tree("foreign").model_copy(update={"reference": _ref(b"x", "other")})

    _succeed(runtime, train, [foreign, _archive("missing"), _tree("kept")], "dsp-1")

    assert set(_stored(registry, train)) == {("final", "kept")}
    assert runtime._tasks[train].status == TaskStatus.DONE


def test_a_report_that_does_not_parse_binds_nothing_but_settles() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    payload = result_payload(runtime._results, train, {"ok": True}, "org")
    payload["artifacts"] = [{"name": "../escape", "kind": "file"}]

    runtime.mark_succeeded(train, "wkr-1", payload, _TS, dispatch_id="dsp-1")

    assert runtime._tasks[train].status == TaskStatus.DONE
    assert _stored(registry, train) == {}


def test_a_conflicting_publication_keeps_the_first() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record = runtime._tasks[train]

    with runtime._lock:
        first = runtime._content_bindings.bind_artifacts_locked(
            record, [_tree()], "dsp-1"
        )
        again = runtime._content_bindings.bind_artifacts_locked(
            record, [_tree(), _tree(data=b"different")], "dsp-2"
        )

    assert len(first) == 1 and again == []
    assert record.artifacts[("final", "final_model")].reference == _ref(b"tree")


def test_a_held_commit_writes_its_artifacts_when_it_lands() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    registry.refuse = 1

    _succeed(runtime, train, [_tree()], "dsp-1")

    assert _stored(registry, train) == {}
    assert runtime._tasks[train].unwritten_artifacts
    runtime._retry_durability(workflow_id)
    assert set(_stored(registry, train)) == {("final", "final_model")}
    assert runtime._tasks[train].unwritten_artifacts == []


def test_a_restart_restores_the_catalog() -> None:
    registry = _registry()
    reader = make_result_reader()
    runtime = _runtime(registry, reader)
    _, (train, use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    _succeed(runtime, train, [_tree()], "dsp-1")

    restored = _runtime(registry, make_result_reader(reader.store))
    assert asyncio.run(restored.rehydrate()) == 1

    assert restored._tasks[train].artifacts == _stored(registry, train)
    assert restored._tasks[train].has_artifacts
    assert not restored._tasks[use].has_artifacts


def test_unregistering_a_workflow_removes_its_catalogs() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    _succeed(runtime, train, [_tree()], "dsp-1")

    registry.unregister_workflows(workflow_id)

    assert raw_control(registry).exists(task_artifacts_key(train)) == 0


@pytest.mark.parametrize("method", ["artifact_bindings", "artifact_bindings_many"])
def test_the_catalog_reads_answer_alike_from_a_thread_or_the_loop(
    method: str,
) -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    _succeed(runtime, train, [_tree()], "dsp-1")
    args: tuple[Any, ...] = (
        (train,) if method == "artifact_bindings" else ([train, use],)
    )

    sync = getattr(registry, method)(*args)
    via_loop = asyncio.run(getattr(registry, f"{method}_async")(*args))

    assert sync == via_loop


def _step(step: int, data: bytes = b"ckpt") -> ArtifactBinding:
    return _tree("checkpoints", data).model_copy(update={"revision": f"step-{step}"})


def _publish(
    runtime: TaskRuntime, task_id: str, binding: ArtifactBinding, dispatch_id: str
) -> bool:
    return runtime.record_artifact_published(
        task_id, "wkr-1", [binding.model_dump(mode="json")], dispatch_id
    )


def test_a_running_task_binds_each_checkpoint_it_publishes() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")

    assert _publish(runtime, train, _step(10), "dsp-1")
    assert not _publish(runtime, train, _step(10), "dsp-1")
    assert not _publish(runtime, train, _step(10, b"retried"), "dsp-1")

    stored = _stored(registry, train)
    assert set(stored) == {("step-10", "checkpoints")}
    assert stored[("step-10", "checkpoints")].reference == _ref(b"ckpt")


def test_a_stale_or_settled_dispatch_binds_no_checkpoint() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")

    assert not _publish(runtime, train, _step(1), "dsp-0")
    _succeed(runtime, train, [], "dsp-1")
    assert not _publish(runtime, train, _step(2), "dsp-1")
    assert _stored(registry, train) == {}


def test_a_cancelling_task_binds_no_checkpoint() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    runtime.cancel_workflow(workflow_id)

    assert runtime._tasks[train].status == TaskStatus.CANCELLING
    assert not _publish(runtime, train, _step(1), "dsp-1")
    assert _stored(registry, train) == {}


def test_a_held_publication_is_not_acknowledged_until_it_lands() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    registry.refuse = 1

    with pytest.raises(TransitionNotDurable), runtime.acknowledging():
        _publish(runtime, train, _step(5), "dsp-1")
    assert _stored(registry, train) == {}

    with runtime.acknowledging():
        assert not _publish(runtime, train, _step(5), "dsp-1")
    assert set(_stored(registry, train)) == {("step-5", "checkpoints")}
    del workflow_id


def test_success_binds_a_checkpoint_whose_report_was_lost() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    _, (train, _use) = _register(runtime)
    record_dispatch(runtime, train, dispatch_id="dsp-1")
    _publish(runtime, train, _step(10), "dsp-1")

    _succeed(runtime, train, [_tree(), _step(10), _step(20)], "dsp-1")

    assert set(_stored(registry, train)) == {
        ("final", "final_model"),
        ("step-10", "checkpoints"),
        ("step-20", "checkpoints"),
    }

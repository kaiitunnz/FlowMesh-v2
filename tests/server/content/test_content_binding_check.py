"""What a task's own bindings entitle its worker to read."""

import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.orchestration.state import (
    AcceptedInput,
    AcceptedInputMember,
    PublicationOutcome,
    ValueRef,
)
from server.task.models import TaskStatus
from server.task.results import ResultUnavailable
from server.task.runtime import TaskRuntime
from server.task.runtime.content_bindings import ContentBindings
from server.task.runtime.dispatch_fence import DispatchFence
from shared.content import (
    TREE_MEDIA_TYPE,
    ContentReference,
    TreeManifest,
    TreeMember,
    reference_for,
)
from shared.harness.adapter import DeliveredOutcome
from shared.outcome import OutcomeManifest
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind

_PREPARED = reference_for("local", b"prepared", media_type="application/json")
_DELIVERED = reference_for("local", b"delivered", media_type="application/json")
_UNRELATED = reference_for("local", b"unrelated", media_type="application/json")


class _Engine:
    """An engine with only the bindings the check reads."""

    def __init__(
        self,
        resolution: Any = None,
        delivered: Any = (),
        settled: dict[str, ContentReference] | None = None,
        child_input: ValueRef | None = None,
        accepted: tuple[AcceptedInput, ...] = (),
    ) -> None:
        self._resolution = resolution
        self._delivered = delivered
        self._settled = settled or {}
        self._child_input = child_input
        self._accepted = accepted

    def input_resolution(self, task_id: str) -> Any:
        return self._resolution

    def episode_context(self, task_id: str) -> tuple[None, Any]:
        return None, self._delivered

    def legacy_task_value(self, task_id: str) -> Any:
        if (reference := self._settled.get(task_id)) is None:
            return None
        return PublicationOutcome.SUCCESS, ValueRef(
            kind="legacy_task_result", legacy_task_id=task_id, content=reference
        )

    def child_input(self, task_id: str) -> ValueRef | None:
        return self._child_input if task_id == "tsk-1" else None

    def accepted_inputs_for_task(self, task_id: str) -> tuple[AcceptedInput, ...]:
        return self._accepted if task_id == "tsk-1" else ()

    def task_inputs(self, task_id: str) -> None:
        return None


class _Manifests:
    def __init__(self, manifests: dict[str, TreeManifest]) -> None:
        self._manifests = manifests
        self.reads = 0

    def tree_members(self, reference: ContentReference) -> frozenset[Any]:
        self.reads += 1
        if (manifest := self._manifests.get(reference.content_digest)) is None:
            raise ResultUnavailable("gone")
        return frozenset(
            member.reference(reference.authorization_scope).identity
            for member in manifest.members
        )


class _NoStore:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the binding check read the store: {name}")


def _record(task_id: str, status: str, **fields: Any) -> Any:
    return SimpleNamespace(
        task_id=task_id,
        assigned_worker=fields.pop("assigned", None),
        dispatch_id=None,
        status=status,
        workflow_id=fields.pop("workflow_id", "wfl-1"),
        org_id="local",
        merged_children=fields.pop("merged_children", None),
        result_skip=None,
        dispatched_artifacts=(),
        dispatched_objects=(),
        **fields,
    )


def _runtime(
    *,
    assigned: str,
    resolution: Any = None,
    delivered: Any = (),
    status: str = TaskStatus.DISPATCHED,
    upstream: dict[str, tuple[str, ContentReference]] | None = None,
    deps: dict[str, set[str]] | None = None,
    child_input: ValueRef | None = None,
    accepted: tuple[AcceptedInput, ...] = (),
    merged_children: list[str] | None = None,
    artifacts: tuple[ArtifactBinding, ...] = (),
    manifests: dict[str, TreeManifest] | None = None,
) -> TaskRuntime:
    """A runtime with only what the binding check reads.

    ``upstream`` maps a task id to its status and the result it settled with.
    """
    runtime = object.__new__(TaskRuntime)
    runtime._lock = threading.RLock()
    tasks: dict[str, Any] = {
        "tsk-1": _record(
            "tsk-1",
            status,
            assigned=assigned,
            merged_children=merged_children,
        )
    }
    settled: dict[str, ContentReference] = {}
    for task_id, (upstream_status, reference) in (upstream or {}).items():
        tasks[task_id] = _record(task_id, upstream_status)
        settled[task_id] = reference
    runtime._tasks = cast(Any, tasks)
    runtime._original_deps = deps or {}
    runtime._engines = cast(
        Any,
        {
            "wfl-1": _Engine(
                resolution,
                delivered,
                settled,
                child_input=child_input,
                accepted=accepted,
            )
        },
    )
    runtime._content_bindings = ContentBindings(
        runtime._tasks,
        runtime._engines,
        runtime._original_deps,
        logging.getLogger("test.content_binding_check"),
    )
    unused = cast(Any, None)
    runtime._results = cast(Any, _Manifests(manifests or {}))
    runtime._fence = DispatchFence(
        unused, unused, unused, unused, unused, runtime._tasks, runtime._engines
    )
    if artifacts and manifests is not None:
        runtime.note_artifact_inputs(
            "tsk-1", artifacts, runtime.artifact_input_objects(artifacts)
        )
    runtime._results = cast(Any, _NoStore())
    return runtime


def test_the_worker_running_a_prepared_task_may_read_its_request() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)


def test_another_worker_may_not_read_it() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-9", _PREPARED)


def test_a_reference_the_task_is_not_bound_to_is_refused() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_an_outcome_the_engine_delivered_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        delivered=(
            DeliveredOutcome(
                call_correlation="c1",
                outcome_ref=OutcomeManifest(content=_DELIVERED),
            ),
        ),
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _DELIVERED)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_an_unknown_task_authorizes_nothing() -> None:
    runtime = _runtime(assigned="wkr-2")
    assert not runtime.content_binding_authorizes("tsk-absent", "wkr-2", _PREPARED)


@pytest.mark.parametrize(
    "status", [TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
def test_the_worker_of_a_settled_task_may_not_read_it(status: str) -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED), status=status
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)


def test_the_worker_winding_down_a_cancelling_task_may_read_it() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        resolution=SimpleNamespace(reference=_PREPARED),
        status=TaskStatus.CANCELLING,
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)


_UPSTREAM = reference_for("local", b"upstream", media_type="application/json")
_ELEMENT = reference_for("local", b"collection", media_type="application/json")


def test_the_settled_result_of_a_transitive_upstream_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        upstream={
            "tsk-mid": (TaskStatus.DONE, _UNRELATED),
            "tsk-root": (TaskStatus.DONE, _UPSTREAM),
        },
        deps={"tsk-1": {"tsk-mid"}, "tsk-mid": {"tsk-root"}},
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-9", _UPSTREAM)


def test_a_result_outside_the_dependency_closure_is_refused() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        upstream={
            "tsk-dep": (TaskStatus.DONE, _UNRELATED),
            "tsk-other": (TaskStatus.DONE, _UPSTREAM),
        },
        deps={"tsk-1": {"tsk-dep"}},
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)


def test_an_upstream_that_has_not_settled_authorizes_nothing() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        upstream={"tsk-dep": (TaskStatus.DISPATCHED, _UPSTREAM)},
        deps={"tsk-1": {"tsk-dep"}},
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)


def test_the_upstream_of_a_task_merged_into_the_dispatch_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        upstream={"tsk-dep": (TaskStatus.DONE, _UPSTREAM)},
        deps={"tsk-child": {"tsk-dep"}},
        merged_children=["tsk-child"],
    )
    runtime._tasks["tsk-child"] = _record("tsk-child", TaskStatus.PENDING)
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)


def test_a_reference_in_another_scope_is_refused() -> None:
    foreign = reference_for("other", b"upstream", media_type="application/json")
    runtime = _runtime(
        assigned="wkr-2",
        upstream={"tsk-dep": (TaskStatus.DONE, foreign)},
        deps={"tsk-1": {"tsk-dep"}},
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", foreign)


def test_the_producer_result_a_fan_out_element_is_frozen_to_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        child_input=ValueRef(
            kind="legacy_task_result",
            legacy_task_id="tsk-producer",
            content=_ELEMENT,
            collection_key="3",
        ),
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _ELEMENT)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_the_producer_result_an_accepted_input_is_frozen_to_may_be_read() -> None:
    accepted = AcceptedInput(
        activation_id="act-1",
        target_port="in",
        members=(
            AcceptedInputMember(
                source_operator_id="producer",
                source_activation_id="act-0",
                outcome=PublicationOutcome.SUCCESS,
                value_ref=ValueRef(
                    kind="legacy_task_result",
                    legacy_task_id="producer",
                    content=_UPSTREAM,
                ),
            ),
        ),
    )
    runtime = _runtime(assigned="wkr-2", accepted=(accepted,))
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-9", _UPSTREAM)


def test_an_accepted_input_frozen_before_it_carried_its_reference_may_be_read() -> None:
    accepted = AcceptedInput(
        activation_id="act-1",
        target_port="in",
        members=(
            AcceptedInputMember(
                source_operator_id="producer",
                source_activation_id="act-0",
                outcome=PublicationOutcome.SUCCESS,
                value_ref=ValueRef(
                    kind="legacy_task_result", legacy_task_id="producer"
                ),
            ),
        ),
    )
    runtime = _runtime(
        assigned="wkr-2",
        accepted=(accepted,),
        upstream={"producer": (TaskStatus.DONE, _UPSTREAM)},
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)


def test_a_merged_child_from_another_workflow_reads_its_own_upstream() -> None:
    runtime = _runtime(assigned="wkr-2", merged_children=["tsk-child"])
    runtime._tasks["tsk-child"] = _record(
        "tsk-child", TaskStatus.PENDING, workflow_id="wfl-2"
    )
    runtime._tasks["tsk-dep"] = _record(
        "tsk-dep",
        TaskStatus.DONE,
        workflow_id="wfl-2",
        result_reference=_UPSTREAM,
        finished_ts=None,
    )
    runtime._original_deps["tsk-child"] = {"tsk-dep"}

    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _UPSTREAM)


_MEMBER = reference_for("local", b"weights")
_MANIFEST = TreeManifest.of(
    [TreeMember("model.safetensors", _MEMBER.content_digest, _MEMBER.size_bytes)]
)
_TREE = ArtifactBinding(
    name="final_model",
    kind=ArtifactKind.TREE,
    reference=reference_for("local", _MANIFEST.encode(), media_type=TREE_MEDIA_TYPE),
    size_bytes=_MEMBER.size_bytes,
)
_FILE = ArtifactBinding(
    name="notes.txt",
    kind=ArtifactKind.FILE,
    reference=reference_for("local", b"notes"),
    size_bytes=5,
)


def test_an_artifact_its_dispatch_names_and_its_trees_members_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        artifacts=(_TREE, _FILE),
        manifests={_TREE.reference.content_digest: _MANIFEST},
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _TREE.reference)
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _FILE.reference)
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _MEMBER)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-9", _MEMBER)


def test_a_tree_whose_manifest_cannot_be_read_names_no_objects_yet() -> None:
    runtime = _runtime(assigned="wkr-2")
    runtime._results = cast(Any, _Manifests({}))
    with pytest.raises(ResultUnavailable):
        runtime.artifact_input_objects((_TREE,))


def test_a_large_trees_members_are_checked_without_reading_the_store() -> None:
    members = [reference_for("local", f"m{index}".encode()) for index in range(10_000)]
    manifest = TreeManifest.of(
        TreeMember(f"f{index}", member.content_digest, member.size_bytes)
        for index, member in enumerate(members)
    )
    tree = ArtifactBinding(
        name="big",
        kind=ArtifactKind.TREE,
        reference=reference_for("local", manifest.encode(), media_type=TREE_MEDIA_TYPE),
        size_bytes=sum(member.size_bytes for member in members),
    )
    runtime = _runtime(
        assigned="wkr-2",
        artifacts=(tree,),
        manifests={tree.reference.content_digest: manifest},
    )
    assert all(
        runtime.content_binding_authorizes("tsk-1", "wkr-2", member)
        for member in members
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_a_settled_task_reads_none_of_its_artifact_inputs() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        artifacts=(_TREE,),
        manifests={_TREE.reference.content_digest: _MANIFEST},
        status=TaskStatus.DONE,
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _MEMBER)

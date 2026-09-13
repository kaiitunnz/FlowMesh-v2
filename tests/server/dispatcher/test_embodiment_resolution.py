"""Tests for resolving a menu node's embodiment inside the dispatch loop."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.dispatcher.embodiment import (
    EmbodimentDecision,
    EmbodimentSnapshot,
    relay_placement_task,
)
from server.task.runtime import TaskRuntime
from server.task.v2.representations.plan import InferenceEmbodimentMenu
from shared.tasks.specs import InferenceEmbodimentKind
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    make_capturing_dispatcher,
)
from tests.server.task.test_v2_embodiment_fence import LOCAL_ELIGIBLE, _runtime
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _worker


class _ForcedSelector:
    """Test-only selector that runs one named embodiment kind."""

    name = "forced"

    def __init__(self, kind: InferenceEmbodimentKind) -> None:
        self._kind = kind

    def __call__(
        self, menu: InferenceEmbodimentMenu, snapshot: EmbodimentSnapshot
    ) -> EmbodimentDecision:
        chosen = next(c for c in menu.candidates if c.kind is self._kind)
        return EmbodimentDecision.select(chosen.alternative_id)


async def _setup(
    primary: str = "resident_served", **kwargs: Any
) -> tuple[CapturingDispatcher, TaskRuntime, str]:
    runtime = _runtime(FakeRegistry())
    _wfl, ids = await _register(runtime, LOCAL_ELIGIBLE.replace("PRIMARY", primary))
    dispatcher = make_capturing_dispatcher(
        runtime=runtime, satisfying_ids=["wkr-1"], **kwargs
    )
    return dispatcher, runtime, ids["gen"]


def _resolve(
    dispatcher: CapturingDispatcher, runtime: TaskRuntime, task_id: str
) -> bool:
    record = runtime.get_record(task_id)
    assert record is not None
    return dispatcher._resolve_embodiment(task_id, record)


@pytest.mark.anyio
async def test_the_primary_embodiment_is_bound_before_placement() -> None:
    dispatcher, runtime, task_id = await _setup(resident_capacity_enabled=True)
    assert _resolve(dispatcher, runtime, task_id) is True

    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED
    assert dispatcher.requeued == []


@pytest.mark.anyio
async def test_an_unplaceable_primary_defers_without_binding_anything() -> None:
    # Resident capacity is off, so the declared resident primary cannot be placed.
    dispatcher, runtime, task_id = await _setup(resident_capacity_enabled=False)
    assert _resolve(dispatcher, runtime, task_id) is False

    # It defers holding no worker: no embodiment bound, no local switch, no retry spent.
    assert runtime.resolved_embodiment(task_id) is None
    assert runtime.service_episode_dispatch(task_id) is None
    [(deferred, kwargs)] = dispatcher.requeued
    assert deferred == task_id
    assert kwargs["count_retry"] is False
    assert "resident_served_infeasible" in kwargs["reason"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind",
    [InferenceEmbodimentKind.RESIDENT_SERVED, InferenceEmbodimentKind.SELF_CONTAINED],
)
async def test_an_injected_selector_can_force_either_legal_candidate(
    kind: InferenceEmbodimentKind,
) -> None:
    dispatcher, runtime, task_id = await _setup(
        embodiment_selector=_ForcedSelector(kind)
    )
    assert _resolve(dispatcher, runtime, task_id) is True

    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None and resolved.kind is kind
    # Routing follows the resolved embodiment, not the leaf's own service binding.
    routes_resident = runtime.service_episode_dispatch(task_id) is not None
    assert routes_resident is (kind is InferenceEmbodimentKind.RESIDENT_SERVED)


@pytest.mark.anyio
async def test_an_uncommitted_embodiment_is_re_resolvable() -> None:
    # Nothing has carried the choice to a worker yet, so a later pass may resolve it
    # again: an embodiment is fixed by its issue, not by the act of recording it.
    dispatcher, runtime, task_id = await _setup(
        embodiment_selector=_ForcedSelector(InferenceEmbodimentKind.SELF_CONTAINED)
    )
    assert _resolve(dispatcher, runtime, task_id) is True
    dispatcher._embodiment_selector = _ForcedSelector(
        InferenceEmbodimentKind.RESIDENT_SERVED
    )

    assert _resolve(dispatcher, runtime, task_id) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind",
    [InferenceEmbodimentKind.RESIDENT_SERVED, InferenceEmbodimentKind.SELF_CONTAINED],
)
async def test_a_delivered_embodiment_is_pinned(
    kind: InferenceEmbodimentKind,
) -> None:
    # Once an attempt has carried the embodiment to a worker it is committed, for a
    # local candidate as much as a resident one — the local candidate never acquires an
    # invocation, so delivery is what fixes it.
    dispatcher, runtime, task_id = await _setup(
        embodiment_selector=_ForcedSelector(kind)
    )
    assert _resolve(dispatcher, runtime, task_id) is True
    runtime.mark_dispatched(task_id, _worker())
    assert runtime.embodiment_pinned(task_id) is True

    other = next(k for k in InferenceEmbodimentKind if k is not kind)
    dispatcher._embodiment_selector = _ForcedSelector(other)
    assert _resolve(dispatcher, runtime, task_id) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None and resolved.kind is kind


@pytest.mark.anyio
async def test_a_permanently_unplaceable_primary_fails_rather_than_hanging() -> None:
    # A defer holds no worker, but it cannot hold forever: past the no-worker grace the
    # task reaches a terminal instead of requeueing for the life of the deployment.
    dispatcher, runtime, task_id = await _setup(
        resident_capacity_enabled=False, grace_sec=0
    )
    assert _resolve(dispatcher, runtime, task_id) is False

    [(failed, message, kwargs)] = dispatcher.failed
    assert failed == task_id
    assert "declared primary" in message
    assert "resident_served_infeasible" in kwargs["payload"]["reason"]
    # It fails rather than switching to the embodiment the author did not declare.
    assert runtime.resolved_embodiment(task_id) is None


@pytest.mark.anyio
async def test_a_task_without_a_menu_passes_straight_through() -> None:
    runtime = TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, None),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("no-menu"),
        secret_vault=cast(Any, None),
    )
    dispatcher = make_capturing_dispatcher(runtime=runtime)
    assert dispatcher._resolve_embodiment("tsk-absent", cast(Any, None)) is True
    assert dispatcher.requeued == []


def _gpu_task(runtime: TaskRuntime, task_id: str) -> Any:
    record = runtime.get_record(task_id)
    assert record is not None
    return record.task


def _declared_gpu(task: Any) -> Any:
    hardware = task.spec.resources.hardware if task.spec.resources else None
    return hardware.gpu if hardware else None


def _declared_cpu(task: Any) -> Any:
    hardware = task.spec.resources.hardware if task.spec.resources else None
    return hardware.cpu if hardware else None


@pytest.mark.anyio
async def test_a_relay_placement_view_drops_only_the_local_accelerator() -> None:
    _dispatcher, runtime, task_id = await _setup()
    task = _gpu_task(runtime, task_id)
    assert _declared_gpu(task) is not None

    relayed = relay_placement_task(task)
    assert _declared_gpu(relayed) is None
    # Every other declared resource still applies, and the original is untouched.
    assert _declared_cpu(relayed) == _declared_cpu(task)
    assert _declared_gpu(task) is not None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind, relaxed",
    [
        (InferenceEmbodimentKind.RESIDENT_SERVED, True),
        (InferenceEmbodimentKind.SELF_CONTAINED, False),
    ],
)
async def test_placement_relaxes_the_accelerator_only_for_a_relaying_embodiment(
    kind: InferenceEmbodimentKind, relaxed: bool
) -> None:
    dispatcher, runtime, task_id = await _setup(
        embodiment_selector=_ForcedSelector(kind)
    )
    assert _resolve(dispatcher, runtime, task_id) is True
    assert dispatcher._relays_only(task_id) is relaxed


@pytest.mark.anyio
async def test_a_task_with_no_resolved_embodiment_places_as_declared() -> None:
    dispatcher, runtime, task_id = await _setup()
    assert dispatcher._relays_only(task_id) is False

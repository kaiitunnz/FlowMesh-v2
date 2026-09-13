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
    PrimaryEmbodimentSelector,
)
from server.task.runtime import TaskRuntime
from server.task.v2.representations.plan import InferenceEmbodimentMenu
from shared.tasks.specs import InferenceEmbodimentKind
from tests.server.dispatcher.helpers import (
    CapturingDispatcher,
    make_capturing_dispatcher,
)
from tests.server.task.test_v2_embodiment_fence import LOCAL_ELIGIBLE, _runtime
from tests.server.task.test_v2_orchestration import FakeRegistry, _register


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


@pytest.mark.anyio
async def test_the_primary_embodiment_is_bound_before_placement() -> None:
    dispatcher, runtime, task_id = await _setup(resident_capacity_enabled=True)
    assert dispatcher._resolve_embodiment(task_id) is True

    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.RESIDENT_SERVED
    assert dispatcher.requeued == []


@pytest.mark.anyio
async def test_an_unplaceable_primary_defers_without_binding_anything() -> None:
    # Resident capacity is off, so the declared resident primary cannot be placed.
    dispatcher, runtime, task_id = await _setup(resident_capacity_enabled=False)
    assert dispatcher._resolve_embodiment(task_id) is False

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
    assert dispatcher._resolve_embodiment(task_id) is True

    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None and resolved.kind is kind
    # Routing follows the resolved embodiment, not the leaf's own service binding.
    routes_resident = runtime.service_episode_dispatch(task_id) is not None
    assert routes_resident is (kind is InferenceEmbodimentKind.RESIDENT_SERVED)


@pytest.mark.anyio
async def test_a_bound_embodiment_is_not_re_resolved_on_a_later_pass() -> None:
    dispatcher, runtime, task_id = await _setup(
        embodiment_selector=_ForcedSelector(InferenceEmbodimentKind.SELF_CONTAINED)
    )
    assert dispatcher._resolve_embodiment(task_id) is True
    dispatcher._embodiment_selector = PrimaryEmbodimentSelector()

    assert dispatcher._resolve_embodiment(task_id) is True
    resolved = runtime.resolved_embodiment(task_id)
    assert resolved is not None
    assert resolved.kind is InferenceEmbodimentKind.SELF_CONTAINED


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
    assert dispatcher._resolve_embodiment("tsk-absent") is True
    assert dispatcher.requeued == []

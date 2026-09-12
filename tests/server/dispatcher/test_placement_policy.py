"""Advisory placement: what a policy may refine, and what it can never reach.

A placement preference is intersected with the pool a dispatch has already narrowed, so
it can only remove a candidate. Owner affinity, a selected-worker hint, and a prior
failure stay hard, and a preference that leaves nothing standing falls back rather than
stranding the task.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig, PolicySurfaceConfig
from server.policy import PlacementContext, PlacementPolicy, build_policy_surface
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.private_state import OwnerFence
from tests.server.dispatcher.helpers import CapturingDispatcher, WorkflowRegistryStub
from tests.server.task.test_v2_orchestration import _NoopSecretVault

_OWNER = OwnerFence(worker_id="wkr-owner", incarnation=7)

_WORKFLOW = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: placement}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""


class _Prefers(PlacementPolicy):
    """Names a fixed worker set, whatever the dispatch narrowed to."""

    def __init__(self, *preferred: str) -> None:
        self._preferred = frozenset(preferred)
        self.seen: list[PlacementContext] = []

    def prefer(self, context: PlacementContext) -> frozenset[str]:
        self.seen.append(context)
        return self._preferred


def _worker(worker_id: str, incarnation: int = _OWNER.incarnation) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="cluster",
        node_id="nde-1",
        node_alias="node",
        incarnation=incarnation,
    )


def _dispatcher(
    *,
    idle_ids: list[str],
    policy: PlacementPolicy | None = None,
    owner_bound: bool = False,
    cached_ids: list[str] | None = None,
) -> tuple[CapturingDispatcher, str, TaskRuntime]:
    surface = build_policy_surface(PolicySurfaceConfig(enabled=True))
    assert surface is not None
    if policy is not None:
        surface = type(surface)(
            lowering=surface.lowering,
            placement=policy,
            state_control=surface.state_control,
        )
    runtime = TaskRuntime(
        cast(Any, WorkflowRegistryStub()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("placement-policy-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        policy=surface,
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )
    task_id = results[0].task_id
    if owner_bound:
        runtime.private_state_owner = mock.Mock(  # type: ignore[method-assign]
            return_value=_OWNER
        )
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker(wid) for wid in idle_ids]
    registry.get_worker.return_value = _worker(_OWNER.worker_id)
    registry.is_worker_stale.return_value = False
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("placement-policy-test"),
        no_worker_grace_sec=0,
    )
    if cached_ids is not None:
        dispatcher._cached_worker_candidates = mock.Mock(  # type: ignore[method-assign]
            return_value=[_worker(wid) for wid in cached_ids]
        )
    return dispatcher, task_id, runtime


def _pools_reaching_selection(
    dispatcher: CapturingDispatcher, task_id: str
) -> list[list[str]]:
    seen: list[list[str]] = []

    def _capture(pool: list[Worker], *args: Any, **kwargs: Any) -> tuple[None, dict]:
        seen.append([worker.id for worker in pool])
        return None, {}

    with mock.patch("server.dispatcher.base.select_worker", _capture):
        dispatcher.dispatch_once(task_id)
    return seen


def test_a_preference_narrows_the_pool() -> None:
    dispatcher, task_id, _ = _dispatcher(
        idle_ids=["wkr-a", "wkr-b"], policy=_Prefers("wkr-b")
    )

    assert _pools_reaching_selection(dispatcher, task_id)[0] == ["wkr-b"]


def test_a_preference_cannot_add_a_worker_outside_the_pool() -> None:
    dispatcher, task_id, _ = _dispatcher(
        idle_ids=["wkr-a"], policy=_Prefers("wkr-elsewhere")
    )

    # The named worker is not idle-eligible, so the preference empties and the pool it
    # was intersected with stands.
    assert _pools_reaching_selection(dispatcher, task_id)[0] == ["wkr-a"]


def test_a_preference_never_softens_owner_affinity() -> None:
    dispatcher, task_id, _ = _dispatcher(
        idle_ids=["wkr-other", _OWNER.worker_id],
        policy=_Prefers("wkr-other"),
        owner_bound=True,
    )

    assert _pools_reaching_selection(dispatcher, task_id)[0] == [_OWNER.worker_id]


def test_an_empty_intersection_falls_back_to_the_cached_preference() -> None:
    dispatcher, task_id, _ = _dispatcher(
        idle_ids=["wkr-a", "wkr-b", "wkr-c"],
        policy=_Prefers("wkr-c"),
        cached_ids=["wkr-a"],
    )

    assert _pools_reaching_selection(dispatcher, task_id)[0] == ["wkr-a"]


def test_both_preferences_compose_when_they_overlap() -> None:
    dispatcher, task_id, _ = _dispatcher(
        idle_ids=["wkr-a", "wkr-b", "wkr-c"],
        policy=_Prefers("wkr-a", "wkr-b"),
        cached_ids=["wkr-b", "wkr-c"],
    )

    assert _pools_reaching_selection(dispatcher, task_id)[0] == ["wkr-b"]


def test_the_policy_reads_the_dispatch_it_is_refining() -> None:
    policy = _Prefers("wkr-a")
    dispatcher, task_id, _ = _dispatcher(idle_ids=["wkr-a", "wkr-b"], policy=policy)

    _pools_reaching_selection(dispatcher, task_id)

    context = policy.seen[0]
    assert context.task_id == task_id
    assert context.candidates == ("wkr-a", "wkr-b")
    assert context.state_generation is None
    assert context.instance_state_holders == frozenset()


def test_no_configured_surface_prefers_nothing() -> None:
    runtime = TaskRuntime(
        cast(Any, WorkflowRegistryStub()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("placement-policy-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )

    assert runtime.placement_preference(results[0].task_id, ["wkr-a"]) == frozenset()

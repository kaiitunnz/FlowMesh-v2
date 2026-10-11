"""Control-plane work is bounded by what a transition does, not by the history or the
open work its workflow has accumulated."""

import time
from collections import Counter
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration.engine.snapshot import SnapshotCodec
from server.orchestration.journal import TrackedDict
from server.orchestration.state import WorkItem
from server.task.models import TaskStatus
from server.task.runtime.commits import TransitionCommitter
from tests.server.orchestration.helpers import chain_bundle, engine
from tests.server.task.test_runtime_control_flow import (
    _ECHO,
    _LOOP_NODES,
    _Run,
    _workflow,
)

_SPAWN_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: kid
            spec: {_ECHO}
          - name: fan
            dependsOn: [step]
            region: {{kind: spawn, child: kid}}
          - name: collect
            dependsOn: [fan]
            region: {{kind: join, completion: all_settled}}
          - name: tally
            dependsOn: [collect]
            spec: {_ECHO}
          - name: route
            dependsOn: [{{node: tally, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_CANCELLING_BODY = f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: work
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {_ECHO}
        edges:
          - from: {{node: work}}
            to: {{node: $return, port: out}}
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: fan
            dependsOn: [step]
            region: {{kind: spawn, child: one}}
          - name: collect
            dependsOn: [fan]
            region: {{kind: join, completion: any, residual: cancel}}
          - name: tally
            dependsOn: [collect]
            spec: {_ECHO}
          - name: route
            dependsOn: [{{node: tally, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_ITERATIONS = 40
_CHILDREN = 2


@pytest.fixture
def walked(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the entries every walk over a keyed ledger collection visits, outside
    the write verifier's own walks."""
    monkeypatch.setattr(SnapshotCodec, "verify_changes", False)
    visits: Counter[str] = Counter()

    def counted(method: Any) -> Any:
        def walk(self: TrackedDict[Any, Any]) -> Iterator[Any]:
            visits[self.name] += len(self)
            return method(self)

        return walk

    for name in ("__iter__", "keys", "values", "items"):
        monkeypatch.setattr(TrackedDict, name, counted(getattr(dict, name)))
    return visits


@pytest.mark.anyio
async def test_a_spawning_loop_body_walks_no_more_ledger_late_than_early(
    walked: Counter[str],
) -> None:
    config = OrchestrationConfig(
        max_loop_iterations=_ITERATIONS + 1,
        max_activations=(_CHILDREN + 6) * _ITERATIONS + 100,
    )
    run = await _Run(config=config).start(_workflow(_LOOP_NODES, _SPAWN_BODY))
    run.run("seed")

    def iteration(last: bool) -> int:
        before = walked.total()
        run.run("step", {"items": [f"x{i}" for i in range(_CHILDREN)]})
        for _ in range(_CHILDREN):
            run.run("kid")
        run.run("tally", {"route": "done" if last else "again"})
        return walked.total() - before

    visits = [iteration(i == _ITERATIONS - 1) for i in range(_ITERATIONS)]
    assert run.settled()
    early, late = visits[5:10], visits[-6:-1]
    assert max(late) <= max(early), (early, late)


def _early_join(children: int) -> str:
    return f"""      - name: planner
        spec: {_ECHO}
      - name: gen
        spec: {_ECHO}
      - name: fanout
        dependsOn: [planner]
        region: {{kind: spawn, child: gen}}
      - name: collect
        dependsOn: [fanout]
        region: {{kind: join, completion: first_k, k: {children}, residual: continue}}
"""


async def _settle_visits(walked: Counter[str], children: int) -> list[int]:
    config = OrchestrationConfig(max_activations=10 * children + 100)
    run = await _Run(config=config).start(_workflow(_early_join(children)))
    run.run("planner", {"items": [f"x{i}" for i in range(children)]})
    visits = []
    for _ in range(children):
        before = walked.total()
        run.run("gen")
        visits.append(walked.total() - before)
    assert run.settled()
    return visits[:-1]


@pytest.mark.anyio
async def test_an_early_join_child_settles_at_a_cost_its_sibling_count_leaves_flat(
    walked: Counter[str],
) -> None:
    few, many = await _settle_visits(walked, 8), await _settle_visits(walked, 40)
    assert max(many) <= max(few), (few, many)


@pytest.mark.anyio
async def test_a_loop_body_cancelling_a_scope_walks_no_more_ledger_late_than_early(
    walked: Counter[str],
) -> None:
    config = OrchestrationConfig(
        max_loop_iterations=_ITERATIONS + 1, max_activations=20 * _ITERATIONS + 100
    )
    run = await _Run(config=config).start(_workflow(_LOOP_NODES, _CANCELLING_BODY))
    run.run("seed")

    def iteration(last: bool) -> int:
        before = walked.total()
        run.run("step", {"items": ["a", "b"]})
        run.run("work")
        # The join released on the first child and cancelled the other.
        run.ready[:] = [t for t in run.ready if run.name(t) != "work"]
        run.run("tally", {"route": "done" if last else "again"})
        return walked.total() - before

    visits = [iteration(i == _ITERATIONS - 1) for i in range(_ITERATIONS)]
    assert run.settled()
    early, late = visits[5:10], visits[-6:-1]
    assert max(late) <= max(early), (early, late)


def _per_unsettled_task_check(open_items: int) -> float:
    eng = engine(chain_bundle())
    for i in range(open_items):
        eng._ledger.add_work_item(
            WorkItem(
                work_item_id=f"wki-{i}",
                activation_id=f"act-{i}",
                operator_id="A",
                legacy_task_id=f"t{i}",
            )
        )
    calls = 4 * open_items
    start = time.perf_counter()
    for _ in range(calls):
        assert eng.has_unsettled_tasks()
    return (time.perf_counter() - start) / calls


def test_an_unsettled_task_check_costs_the_same_however_many_items_are_open() -> None:
    few, many = _per_unsettled_task_check(1_000), _per_unsettled_task_check(100_000)
    assert many < 5 * few, (few, many)


_comparisons = Counter[str]()


class _CountedId(str):
    """A task id that counts the equality checks it takes part in."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        _comparisons["eq"] += 1
        return str.__eq__(self, other)


def _records_commit_comparisons(size: int) -> int:
    tasks = {
        f"t{i}": SimpleNamespace(
            workflow_id="w", status=TaskStatus.DONE, residual_cancel=False
        )
        for i in range(size)
    }
    committer = SimpleNamespace(
        _tasks=tasks,
        _engines={},
        _records_locked=lambda *ids: [],
        _sched_locked=lambda workflow_id: None,
        _workflow_registry=SimpleNamespace(commit_transition=lambda *a, **k: None),
        _after_records_locked=lambda *a: None,
    )
    ids = [_CountedId(task_id) for task_id in tasks]
    moves = [_CountedId(task_id) for task_id in tasks]
    _comparisons.clear()
    TransitionCommitter._commit_records_raw(
        cast(Any, committer), "w", ids, moves, sched=False
    )
    return _comparisons["eq"]


def test_a_records_commit_compares_each_task_a_bounded_number_of_times() -> None:
    small, large = _records_commit_comparisons(500), _records_commit_comparisons(2_000)
    assert large <= 4 * small + 100, (small, large)

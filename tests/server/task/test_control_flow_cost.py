"""A loop's per-iteration control-plane work is bounded by what the iteration does, not
by the history the loop has accumulated."""

from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest

from server.config import OrchestrationConfig
from server.orchestration.engine.snapshot import SnapshotCodec
from server.orchestration.journal import TrackedDict
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

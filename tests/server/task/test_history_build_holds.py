"""Building a workflow's stored history holds the runtime lock a bounded chunk at a
time, releasing it between chunks."""

import threading
import time
from typing import Any

import pytest

from server.orchestration.engine import OrchestrationEngine
from server.task.runtime import TaskRuntime, facade
from tests.server.task.test_settled_history import (
    _certified,
    _register,
    _registry,
    _resident,
    _runtime,
    _succeed_ready,
)

_LEAVES = 6
_CHUNK = 2

_PUBLISHED = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: published}
spec:
  graph:
    nodes:
""" + "".join(
    f"""      - name: n{i}
        spec:
          taskType: echo
          data: {{type: list, items: [x]}}
          v2: {{result: {{visibility: published}}}}
""" for i in range(_LEAVES)
)


class _Clock:
    """The facade's ``time``, recording each yield between holds."""

    def __init__(self, on_yield: Any) -> None:
        self._on_yield = on_yield

    def sleep(self, seconds: float) -> None:
        self._on_yield()
        time.sleep(seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def test_the_history_is_built_in_bounded_holds_of_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id = _register(runtime, _PUBLISHED)
    assert len(_succeed_ready(runtime)) == _LEAVES
    events: list[tuple[str, bool]] = []

    def held() -> bool:
        """Whether another thread is kept from taking the runtime lock."""
        taken: list[bool] = []

        def take() -> None:
            if runtime._lock.acquire(blocking=False):
                runtime._lock.release()
                taken.append(True)

        probe = threading.Thread(target=take)
        probe.start()
        probe.join()
        return not taken

    derived = TaskRuntime._derived_locked
    publication = OrchestrationEngine.output_publication

    def derived_spy(self: TaskRuntime, *args: Any) -> Any:
        events.append(("task", held()))
        return derived(self, *args)

    def publication_spy(self: OrchestrationEngine, *args: Any) -> Any:
        events.append(("output", held()))
        return publication(self, *args)

    monkeypatch.setattr(facade, "_HISTORY_BUILD_CHUNK", _CHUNK)
    monkeypatch.setattr(facade, "time", _Clock(lambda: events.append(("gap", held()))))
    monkeypatch.setattr(TaskRuntime, "_derived_locked", derived_spy)
    monkeypatch.setattr(OrchestrationEngine, "output_publication", publication_spy)

    runtime.settled_history.drain()

    assert _certified(registry, workflow_id)
    assert not _resident(runtime, workflow_id)
    for kind in ("task", "output"):
        built = [event for event in events if event[0] in (kind, "gap")]
        assert [locked for name, locked in built if name == kind] == [True] * _LEAVES
        holds: list[int] = [0]
        for name, locked in built:
            if name == "gap":
                assert not locked
                holds.append(0)
            else:
                holds[-1] += 1
        assert max(holds) == _CHUNK
        assert sum(1 for count in holds if count) == _LEAVES // _CHUNK

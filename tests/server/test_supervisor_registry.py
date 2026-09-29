"""WorkerRegistry behaviour and thread-safety under concurrent access."""

import threading
from collections.abc import Callable
from typing import cast

import pytest

from server.supervisor.adapters.base import WorkerAdapter, WorkerTokenType
from server.supervisor.registry import WorkerRegistry


class _FakeAdapter:
    def __init__(self, token: str, name: str) -> None:
        self.token = cast(WorkerTokenType, token)
        self.name = name


def _adapter(token: str, name: str) -> WorkerAdapter:
    return cast(WorkerAdapter, _FakeAdapter(token, name))


def test_add_get_pop_roundtrip() -> None:
    registry = WorkerRegistry()
    worker = _adapter("tok-1", "worker-1")

    registry.add(worker)
    assert registry.try_get(cast(WorkerTokenType, "tok-1")) is worker
    assert registry.try_get_by_name("worker-1") is worker
    assert registry.all_workers() == [worker]

    registry.set_worker_id(cast(WorkerTokenType, "tok-1"), "wrk-1")
    assert registry.get_worker_id(cast(WorkerTokenType, "tok-1")) == "wrk-1"

    popped = registry.try_pop(cast(WorkerTokenType, "tok-1"))
    assert popped is worker
    assert registry.all_workers() == []
    assert registry.try_get(cast(WorkerTokenType, "tok-1")) is None
    assert registry.get_worker_id(cast(WorkerTokenType, "tok-1")) is None


def test_add_rejects_duplicate_token_and_name() -> None:
    registry = WorkerRegistry()
    registry.add(_adapter("tok-1", "worker-1"))

    with pytest.raises(ValueError, match="token"):
        registry.add(_adapter("tok-1", "worker-2"))
    with pytest.raises(ValueError, match="name"):
        registry.add(_adapter("tok-2", "worker-1"))


class _YieldingName(str):
    """A name whose second hash (``add``'s insert, after its duplicate check)
    waits for ``resume`` so another thread can act in between."""

    _resume: threading.Event
    _hashes: int

    def __new__(cls, value: str, resume: threading.Event) -> "_YieldingName":
        name = super().__new__(cls, value)
        name._resume = resume
        name._hashes = 0
        return name

    def __hash__(self) -> int:
        self._hashes += 1
        if self._hashes == 2:
            self._resume.wait(timeout=0.2)
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other)


def test_add_is_atomic_against_a_concurrent_add_of_the_same_name() -> None:
    registry = WorkerRegistry()
    other_done = threading.Event()
    results: dict[str, bool] = {}

    def add(token: str, name: str) -> None:
        try:
            registry.add(_adapter(token, name))
            results[token] = True
        except ValueError:
            results[token] = False

    first = threading.Thread(
        target=add, args=("tok-1", _YieldingName("worker", other_done))
    )
    first.start()

    def add_then_signal() -> None:
        add("tok-2", "worker")
        other_done.set()

    second = threading.Thread(target=add_then_signal)
    second.start()
    first.join()
    second.join()

    assert sorted(results.values()) == [False, True]
    assert len(registry.all_workers()) == 1


def _tok(token: str) -> WorkerTokenType:
    return cast(WorkerTokenType, token)


def _registry_with_release_log() -> tuple[WorkerRegistry, list[str]]:
    released: list[str] = []
    return WorkerRegistry(on_worker_id_released=released.append), released


def test_release_fires_when_binding_is_replaced() -> None:
    registry, released = _registry_with_release_log()

    registry.set_worker_id(_tok("tok-1"), "wkr-1")
    registry.set_worker_id(_tok("tok-1"), "wkr-1")
    assert released == []

    registry.set_worker_id(_tok("tok-1"), "wkr-2")
    assert released == ["wkr-1"]
    assert registry.get_worker_id(_tok("tok-1")) == "wkr-2"


_POPS = pytest.mark.parametrize(
    "pop",
    [
        lambda r: r.pop(_tok("tok-1")),
        lambda r: r.try_pop(_tok("tok-1")),
        lambda r: r.pop_by_name("worker-1"),
        lambda r: r.try_pop_by_name("worker-1"),
    ],
    ids=["pop", "try_pop", "pop_by_name", "try_pop_by_name"],
)


@_POPS
def test_release_fires_for_each_pop_of_a_bound_token(
    pop: Callable[[WorkerRegistry], object],
) -> None:
    registry, released = _registry_with_release_log()
    registry.add(_adapter("tok-1", "worker-1"))
    registry.set_worker_id(_tok("tok-1"), "wkr-1")

    pop(registry)

    assert released == ["wkr-1"]


@_POPS
def test_release_skips_pop_of_an_unbound_token(
    pop: Callable[[WorkerRegistry], object],
) -> None:
    registry, released = _registry_with_release_log()
    registry.add(_adapter("tok-1", "worker-1"))

    pop(registry)

    assert released == []


def test_release_skips_missing_tokens() -> None:
    registry, released = _registry_with_release_log()

    assert registry.try_pop(_tok("tok-missing")) is None
    assert registry.try_pop_by_name("worker-missing") is None
    with pytest.raises(KeyError):
        registry.pop(_tok("tok-missing"))

    assert released == []


def test_release_fires_once_per_bound_id_on_clear() -> None:
    registry, released = _registry_with_release_log()
    for i in range(3):
        registry.add(_adapter(f"tok-{i}", f"worker-{i}"))
    registry.set_worker_id(_tok("tok-0"), "wkr-0")
    registry.set_worker_id(_tok("tok-1"), "wkr-1")

    registry.clear()

    assert sorted(released) == ["wkr-0", "wkr-1"]
    assert registry.get_worker_id(_tok("tok-0")) is None


def test_release_callback_runs_outside_the_lock() -> None:
    """A callback that re-enters the registry must not deadlock."""
    seen: list[str | None] = []
    registry = WorkerRegistry(
        on_worker_id_released=lambda _: seen.append(
            registry.get_worker_id(_tok("tok-1"))
        )
    )
    registry.add(_adapter("tok-1", "worker-1"))
    registry.set_worker_id(_tok("tok-1"), "wkr-1")

    def mutate() -> None:
        registry.set_worker_id(_tok("tok-1"), "wkr-2")
        registry.try_pop(_tok("tok-1"))

    thread = threading.Thread(target=mutate, daemon=True)
    thread.start()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert seen == ["wkr-2", None]

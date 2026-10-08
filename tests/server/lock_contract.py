"""Check that the task runtime's lock-bound helpers run under its lock, and that what
it delivers after a commit runs off it.

Every ``*_locked`` method of a runtime component, each listed unsuffixed helper, and
every public method of an orchestration engine a runtime holds must run with that
runtime's lock held; each listed delivery must run without it. A breach from runtime
source fails the test that caused it. A test may call a helper directly, so a breach
it makes itself is only recorded.
"""

import functools
import inspect
import sys
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Any

from server.orchestration import OrchestrationEngine
from server.task.runtime import (
    after_commit,
    agent_inputs,
    commits,
    content_bindings,
    dispatch_fence,
    episode_dispatch,
    facade,
    fanout,
    input_checks,
    materialized_records,
    mediated_ops,
    merges,
    record_failures,
    reports,
    reservations,
    resident_tasks,
    scheduling,
    static_dag,
)

_MODULES = (
    after_commit,
    agent_inputs,
    commits,
    content_bindings,
    dispatch_fence,
    episode_dispatch,
    facade,
    fanout,
    input_checks,
    materialized_records,
    mediated_ops,
    merges,
    record_failures,
    reports,
    reservations,
    resident_tasks,
    scheduling,
    static_dag,
)
_MODULE_NAMES = frozenset(module.__name__ for module in _MODULES)

# Helpers that read or move shared runtime state without the ``_locked`` suffix.
_UNSUFFIXED = {
    "EpochFrontier": (
        "set_frontier",
        "complete_epoch",
        "drop_frontier",
        "drop_epochs",
        "forget_workflow",
        "forget_task",
    ),
    "ReadyQueue": ("merge_bucket_remove", "set_merge_key", "forget_merge_key"),
    "TaskMerges": ("take_children", "take_parent", "drop_merged_child"),
    "AfterCommitActions": ("take_ready", "has_failed"),
    "MediatedOperations": (
        "take_stale_ops",
        "record_issued_op",
        "take_settled",
        "take_overdue",
        "discard_op",
        "pending_for_worker",
        "drop_worker_ops",
    ),
    "InputChecks": ("drop_check",),
    "WorkerReservations": ("take_ended", "requeue_ended"),
    "DispatchFence": ("take_publish", "mark_publish_lost", "remember_return"),
    "TransitionCommitter": ("durable", "drop_unacknowledged"),
}

# What the runtime delivers to workers and resident control once a commit is durable.
_OFF_LOCK = {
    "TaskRuntime": ("_act_after_commit", "_deliver"),
    "WorkerReservations": ("release_workers",),
    "MediatedOperations": (
        "reap_mediated_op",
        "relay_resident_reap",
        "release_resident_credit",
    ),
}


@dataclass(frozen=True)
class Trip:
    kind: str
    name: str
    site: str
    from_source: bool


_runtimes: "weakref.WeakSet[Any]" = weakref.WeakSet()
_owner: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()
_trips: list[Trip] = []
_trips_lock = threading.Lock()
_HERE = __file__


def _caller() -> tuple[str, bool]:
    frame: FrameType | None = sys._getframe(3)
    while frame is not None and frame.f_code.co_filename == _HERE:
        frame = frame.f_back
    if frame is None:
        return "?", False
    path = frame.f_code.co_filename
    return f"{path}:{frame.f_lineno} {frame.f_code.co_name}", "/src/" in path


def _trip(kind: str, name: str) -> None:
    site, from_source = _caller()
    with _trips_lock:
        _trips.append(Trip(kind, name, site, from_source))


def _lock_of(obj: Any) -> Any:
    if (lock := _owner.get(obj)) is not None:
        return lock
    if isinstance(obj, OrchestrationEngine):
        for runtime in list(_runtimes):
            if any(engine is obj for engine in list(runtime._engines.values())):
                return runtime._lock
    return None


def _wrap(cls: type, name: str, breach: Callable[[Any], bool], kind: str) -> None:
    fn = cls.__dict__[name]
    label = f"{cls.__name__}.{name}"

    @functools.wraps(fn)
    def checked(self: Any, *args: Any, **kwargs: Any) -> Any:
        if (lock := _lock_of(self)) is not None and breach(lock):
            _trip(kind, label)
        return fn(self, *args, **kwargs)

    setattr(cls, name, checked)


def _requires(cls: type, name: str) -> None:
    _wrap(cls, name, lambda lock: not lock._is_owned(), "unlocked")


def _forbids(cls: type, name: str) -> None:
    _wrap(cls, name, lambda lock: lock._is_owned(), "locked")


def _bind(runtime: Any) -> None:
    _runtimes.add(runtime)
    _owner[runtime] = runtime._lock
    for value in vars(runtime).values():
        if type(value).__module__ in _MODULE_NAMES:
            _owner[value] = runtime._lock


def install() -> None:
    for module in _MODULES:
        for cls in vars(module).values():
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            for name, attr in list(vars(cls).items()):
                if inspect.isfunction(attr) and name.endswith("_locked"):
                    _requires(cls, name)
            for name in _UNSUFFIXED.get(cls.__name__, ()):
                _requires(cls, name)
            for name in _OFF_LOCK.get(cls.__name__, ()):
                _forbids(cls, name)
    for name, attr in list(vars(OrchestrationEngine).items()):
        if inspect.isfunction(attr) and not name.startswith("_"):
            _requires(OrchestrationEngine, name)
    init = facade.TaskRuntime.__init__

    @functools.wraps(init)
    def bound_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        _bind(self)

    facade.TaskRuntime.__init__ = bound_init  # type: ignore[method-assign]


def take_trips() -> list[Trip]:
    with _trips_lock:
        taken = list(_trips)
        _trips.clear()
    return taken


@contextmanager
def recorded() -> Iterator[list[Trip]]:
    """Collect the trips made inside the block, so they fail no test."""
    take_trips()
    trips: list[Trip] = []
    try:
        yield trips
    finally:
        trips.extend(take_trips())

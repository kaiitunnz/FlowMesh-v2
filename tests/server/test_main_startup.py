"""Root startup rebuilds durable state in a credit-safe order: resident-capacity control
binds and loads its claim store before the runtime re-drives suspended boundaries."""

import asyncio
from typing import Any

from server.main import _rehydrate_root_state


class _Recorder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls


class _Runtime(_Recorder):
    async def rehydrate(self) -> None:
        self.calls.append("runtime.rehydrate")


class _Registry(_Recorder):
    async def load_snapshot_async(self) -> Any:
        self.calls.append("load_snapshot")
        return object()  # a non-None snapshot


class _Control(_Recorder):
    def bind_loop(self, loop: Any) -> None:
        self.calls.append("bind_loop")

    def rehydrate(self, snapshot: Any) -> None:
        self.calls.append("control.rehydrate")

    def start(self) -> None:
        self.calls.append("start")


def test_root_startup_loads_resident_capacity_before_the_runtime() -> None:
    calls: list[str] = []
    runtime, control, registry = _Runtime(calls), _Control(calls), _Registry(calls)
    asyncio.run(
        _rehydrate_root_state(runtime, control, registry)  # type: ignore[arg-type]
    )
    # The BLOCKER guard: the claim store is bound and loaded before the runtime
    # re-drives suspended boundaries (a resident boundary would otherwise terminalize
    # against an empty store), and the idle sweep starts only after the store rebuilds.
    assert calls == [
        "bind_loop",
        "load_snapshot",
        "control.rehydrate",
        "runtime.rehydrate",
        "start",
    ]

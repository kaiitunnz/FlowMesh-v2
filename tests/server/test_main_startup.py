"""Root startup rebuilds durable state in a credit-safe order: resident-capacity control
binds and loads its claim store before the runtime re-drives suspended boundaries."""

import asyncio
from typing import Any

from server.startup import rehydrate_root_state
from tests.server.runtime_helpers import never_closed


class _Recorder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls


class _Runtime(_Recorder):
    async def restart(self, *args: Any) -> None:
        self.calls.append("runtime.restart")

    def release_ended_reservations(self) -> None:
        self.calls.append("runtime.release")

    def live_resident_task_ids(self) -> set[str]:
        return set()

    def resident_invocation_completed(
        self, workflow_id: str, invocation_id: str
    ) -> None:
        return None


class _Registry(_Recorder):
    def __init__(self, calls: list[str], snapshot: Any = None) -> None:
        super().__init__(calls)
        self.snapshot = object() if snapshot is None else snapshot

    async def load_snapshot_async(self) -> Any:
        self.calls.append("load_snapshot")
        return self.snapshot


class _Control(_Recorder):
    def bind_loop(self, loop: Any) -> None:
        self.calls.append("bind_loop")

    def rehydrate(self, snapshot: Any) -> None:
        self.calls.append("control.rehydrate")

    def claims_terminal(self, invocation_ids: Any) -> bool:
        return True

    def reattach_replicas(self, live_serve_tasks: Any) -> None:
        self.calls.append("control.reattach")

    def recover_retirements(self) -> None:
        self.calls.append("control.recover")

    def reconcile_workflow_terminals(self, completed: Any) -> None:
        self.calls.append("control.reconcile")

    def start(self) -> None:
        self.calls.append("start")


class _GatedServe(_Recorder):
    def reconcile_terminals(self) -> None:
        self.calls.append("serve.reconcile")


def test_root_startup_loads_resident_capacity_before_the_runtime() -> None:
    calls: list[str] = []
    runtime, control, registry = _Runtime(calls), _Control(calls), _Registry(calls)
    asyncio.run(
        rehydrate_root_state(
            runtime, control, registry, None, never_closed  # type: ignore[arg-type]
        )
    )
    # The claim store is bound and loaded before the runtime
    # re-drives suspended boundaries (a resident boundary would otherwise terminalize
    # against an empty store), restored replicas re-attach once the runtime has restored
    # their serve tasks, invocations consumed before the restart finish their cleanup
    # once the replicas they name are re-attached, the claims the restored ledger
    # settled release before the idle sweep starts, and it starts only after the store
    # rebuilds.
    assert calls == [
        "bind_loop",
        "load_snapshot",
        "control.rehydrate",
        "runtime.restart",
        "control.reattach",
        "control.recover",
        "runtime.release",
        "control.reconcile",
        "start",
    ]


def test_root_startup_settles_serve_requests_before_admission_opens() -> None:
    calls: list[str] = []
    runtime, control, registry = _Runtime(calls), _Control(calls), _Registry(calls)
    gated_serve = _GatedServe(calls)
    asyncio.run(
        rehydrate_root_state(
            runtime, control, registry, gated_serve, never_closed  # type: ignore[arg-type]
        )
    )
    # The serve requests the restart ended settle once the runtime has restored the
    # serve tasks whose workers reap them, and before re-attaching replicas opens
    # admission.
    assert calls == [
        "bind_loop",
        "load_snapshot",
        "control.rehydrate",
        "runtime.restart",
        "serve.reconcile",
        "control.reattach",
        "control.recover",
        "runtime.release",
        "control.reconcile",
        "start",
    ]

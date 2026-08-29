"""The Codex app-server adapter maps the harness contract and recovers on the rollout.

A fake app-server scripts turn/item events so the binding runs without a live Codex: a
facade tool defers as a generic boundary, an injected outcome resumes the turn, and a
finished turn completes. The load-bearing part is crash recovery — across an app-server
loss against the same persisted rollout, a held single-forward facade call re-drives and
its mediated effect runs exactly once, gated by the fabric idempotency key rather than a
Codex-local call id.
"""

import json
from collections import defaultdict
from collections.abc import Sequence

import pytest

from shared.harness import (
    BoundaryEventKind,
    DeliveredOutcome,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from worker.harness.codex import (
    CodexAppServerHarnessAdapter,
    CodexEvent,
    CodexInjectItem,
    RealCodexAppServerTransport,
)


class FakeCodexAppServer:
    """A persisted rollout: it scripts blocks and dedupes injected effects by key."""

    def __init__(self, blocks: list[dict]) -> None:
        self._blocks = blocks
        self._thread_id = "thr-1"
        self.cursor = 0
        self.committed_keys: set[str] = set()
        self.execution_count: dict[str, int] = defaultdict(int)
        self._call_seq = 0
        self.resumed = 0
        self.cancelled = 0
        self.received_keys: list[str | None] = []

    def thread_start(self) -> str:
        return self._thread_id

    def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
        assert thread_id == self._thread_id
        self.resumed += 1

    def thread_inject_items(
        self, thread_id: str, items: Sequence[CodexInjectItem]
    ) -> None:
        for item in items:
            self.received_keys.append(item.idempotency_key)
            key = item.idempotency_key
            if key is not None and key not in self.committed_keys:
                self.committed_keys.add(key)
                self.execution_count[item.call_correlation] += 1
                self.cursor += 1

    def turn_start(self, thread_id: str) -> str:
        return f"turn-{self.cursor}"

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        block = self._blocks[self.cursor]
        if block["kind"] == "facade_call":
            self._call_seq += 1
            return CodexEvent(
                kind="facade_call",
                call_id=f"call-{self._call_seq}",
                tool=block["tool"],
                interface=block.get("interface"),
                region=block.get("region"),
                arguments=block.get("arguments"),
            )
        return CodexEvent(kind=block["kind"], value=block.get("value"))

    def cancel(self, thread_id: str) -> None:
        self.cancelled += 1


def _settle(result: HarnessResult, value: str = "ok") -> DeliveredOutcome:
    """The fabric's role: record the boundary and inject its keyed outcome."""
    assert result.request is not None and result.request.call_correlation is not None
    corr = result.request.call_correlation
    return DeliveredOutcome(
        call_correlation=corr,
        idempotency_key=f"idm-{corr}",
        kind=OutcomeKind.RESULT,
        value=value,
    )


_LIFECYCLE = [
    {"kind": "facade_call", "tool": "spawn_agent", "region": "reviewer"},
    {"kind": "facade_call", "tool": "ask", "interface": "model", "arguments": "q"},
    {"kind": "completed", "value": "done"},
]


def test_backend_key_pins_the_version() -> None:
    adapter = CodexAppServerHarnessAdapter(FakeCodexAppServer([]), "v1")
    key = adapter.backend_key()
    assert key.backend == "codex" and key.version == "v1"


def test_bypass_paths_are_disabled() -> None:
    assert CodexAppServerHarnessAdapter(FakeCodexAppServer([])).bypass_disabled()


def test_full_lifecycle_maps_onto_the_generic_contract() -> None:
    fake = FakeCodexAppServer(_LIFECYCLE)
    adapter = CodexAppServerHarnessAdapter(fake)

    spawn = adapter.start("a", capsule=None, outcomes=[])
    assert spawn.kind is HarnessResultKind.BOUNDARY
    assert spawn.request is not None
    assert spawn.request.kind is BoundaryEventKind.SPAWN
    assert spawn.request.child_region_ref == "reviewer"

    invocation = adapter.start("a", capsule=spawn.capsule, outcomes=[_settle(spawn)])
    assert invocation.request is not None
    assert invocation.request.kind is BoundaryEventKind.INVOCATION
    assert invocation.request.interface == "model"

    done = adapter.start(
        "a", capsule=invocation.capsule, outcomes=[_settle(invocation, "answer")]
    )
    assert done.kind is HarnessResultKind.COMPLETION and done.value == "done"
    # Each mediated facade effect committed exactly once.
    assert dict(fake.execution_count) == {"thr-1:0": 1, "thr-1:1": 1}


def test_a_turn_error_fails_the_episode() -> None:
    fake = FakeCodexAppServer([{"kind": "error", "value": "boom"}])
    result = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.FAILURE and result.error == "boom"


def test_multi_facade_block_is_rejected() -> None:
    # An outstanding spawn followed by a different facade kind before it resolves is a
    # multi-facade block the single-forward binding does not lift.
    fake = FakeCodexAppServer(
        [{"kind": "facade_call", "tool": "spawn_agent", "region": "r"}]
    )
    adapter = CodexAppServerHarnessAdapter(fake)
    first = adapter.start("a", capsule=None, outcomes=[])
    # Re-drive without resolving, but the rollout now scripts a different facade kind.
    fake._blocks[0] = {"kind": "facade_call", "tool": "ask", "interface": "model"}
    with pytest.raises(ValueError, match="multi-facade"):
        adapter.start("a", capsule=first.capsule, outcomes=[])


_HELD = [
    {"kind": "facade_call", "tool": "spawn_agent", "region": "reviewer"},
    {"kind": "completed", "value": "final"},
]


def test_crash_before_issue_redrives_on_the_same_rollout() -> None:
    fake = FakeCodexAppServer(_HELD)
    # The first attempt defers the facade call but the boundary is lost before issue.
    CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    # A fresh adapter re-drives from the start against the same rollout.
    adapter = CodexAppServerHarnessAdapter(fake)
    redriven = adapter.start("a", capsule=None, outcomes=[])
    assert redriven.request is not None
    corr = redriven.request.call_correlation
    assert corr is not None
    done = adapter.start("a", capsule=redriven.capsule, outcomes=[_settle(redriven)])
    assert done.kind is HarnessResultKind.COMPLETION
    assert fake.execution_count[corr] == 1


def test_crash_after_issue_before_injection_keeps_a_stable_correlation() -> None:
    fake = FakeCodexAppServer(_HELD)
    first = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert first.request is not None
    corr = first.request.call_correlation
    assert corr is not None
    codex_call_before = _codex_call_id(first.capsule)

    # App-server loss before the outcome injects: a fresh adapter resumes and re-drives.
    adapter = CodexAppServerHarnessAdapter(fake)
    redriven = adapter.start("a", capsule=first.capsule, outcomes=[])
    assert redriven.request is not None
    # The correlation is stable across the re-drive even though Codex reissued the call.
    assert redriven.request.call_correlation == corr
    assert _codex_call_id(redriven.capsule) != codex_call_before
    assert fake.resumed >= 1

    done = adapter.start("a", capsule=redriven.capsule, outcomes=[_settle(redriven)])
    assert done.kind is HarnessResultKind.COMPLETION
    assert fake.execution_count[corr] == 1  # the mediated effect ran exactly once


def test_crash_after_injection_before_terminal_does_not_reexecute() -> None:
    fake = FakeCodexAppServer(_HELD)
    first = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert first.request is not None
    corr = first.request.call_correlation
    assert corr is not None
    outcome = _settle(first)

    # The outcome injects and the turn completes, but the completion is lost to a crash.
    CodexAppServerHarnessAdapter(fake).start(
        "a", capsule=first.capsule, outcomes=[outcome]
    )
    assert fake.execution_count[corr] == 1

    # Recovery re-delivers the same keyed outcome against the same rollout; the
    # effect is deduped, not re-executed, and the episode still completes.
    adapter = CodexAppServerHarnessAdapter(fake)
    done = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
    assert done.kind is HarnessResultKind.COMPLETION and done.value == "final"
    assert fake.execution_count[corr] == 1


def test_a_recommitted_outcome_is_not_reinjected() -> None:
    # A re-dispatch re-ships the same pending outcome; the adapter dedupes injection by
    # the committed fabric key, so it is not passed to the app-server twice.
    fake = FakeCodexAppServer(_LIFECYCLE)
    adapter = CodexAppServerHarnessAdapter(fake)
    first = adapter.start("a", capsule=None, outcomes=[])
    outcome = _settle(first)
    resumed = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
    # Re-ship the identical keyed outcome against the advanced capsule.
    adapter.start("a", capsule=resumed.capsule, outcomes=[outcome])
    assert fake.received_keys.count(outcome.idempotency_key) == 1


def test_real_transport_is_an_unbound_seam() -> None:
    with pytest.raises(NotImplementedError):
        RealCodexAppServerTransport().thread_start()


def _codex_call_id(capsule: HarnessCapsule | None) -> str | None:

    assert capsule is not None
    outstanding = json.loads(capsule.blob)["outstanding"]
    return None if outstanding is None else outstanding["codex_call_id"]

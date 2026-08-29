"""The Codex app-server adapter maps the harness contract and recovers on the rollout.

A fake app-server scripts terminal turns so the binding runs without a live Codex: each
turn completes or fails, and a delivered outcome injects back and resumes the rollout.
A facade originates at the gateway, not the adapter, so the adapter never observes one.
The load-bearing part is crash recovery — across an app-server loss against the same
persisted rollout, a re-delivered outcome injects at most once, gated by the committed
fabric idempotency key rather than a Codex-local call id.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pytest

from shared.harness import (
    DeliveredOutcome,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks.specs import AgentSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors.harness.codex import (
    CodexAppServerHarnessAdapter,
    CodexEvent,
    CodexInjectItem,
    _agent_task,
    _isolated_codex_home,
)


class FakeCodexAppServer:
    """A persisted rollout: it scripts terminal turns and dedupes injects by key."""

    def __init__(self, blocks: list[dict]) -> None:
        self._blocks = blocks
        self._thread_id = "thr-1"
        self.cursor = 0
        self.committed_keys: set[str] = set()
        self.execution_count: dict[str, int] = defaultdict(int)
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
                # The persisted rollout records each keyed effect once, so a re-inject
                # from a stale capsule is deduped here even if the adapter re-ships it.
                self.committed_keys.add(key)
                self.execution_count[item.call_correlation] += 1

    def turn_start(self, thread_id: str) -> str:
        return f"turn-{self.cursor}"

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        block = self._blocks[min(self.cursor, len(self._blocks) - 1)]
        self.cursor += 1
        return CodexEvent(kind=block["kind"], value=block.get("value"))

    def cancel(self, thread_id: str) -> None:
        self.cancelled += 1


def _outcome(corr: str, value: str = "child") -> DeliveredOutcome:
    """A settled child outcome the fabric delivers back at its originating call."""
    return DeliveredOutcome(
        call_correlation=corr,
        idempotency_key=f"idm-{corr}",
        kind=OutcomeKind.RESULT,
        value=value,
    )


def test_backend_key_pins_the_version() -> None:
    adapter = CodexAppServerHarnessAdapter(FakeCodexAppServer([]), "v1")
    key = adapter.backend_key()
    assert key.backend == "codex" and key.version == "v1"


def test_bypass_paths_are_disabled() -> None:
    assert CodexAppServerHarnessAdapter(FakeCodexAppServer([])).bypass_disabled()


def test_a_completed_turn_completes_the_episode() -> None:
    fake = FakeCodexAppServer([{"kind": "completed", "value": "done"}])
    result = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.COMPLETION and result.value == "done"


def test_a_turn_error_fails_the_episode() -> None:
    fake = FakeCodexAppServer([{"kind": "error", "value": "boom"}])
    result = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.FAILURE and result.error == "boom"


def test_a_delivered_outcome_injects_and_resumes_the_rollout() -> None:
    fake = FakeCodexAppServer(
        [
            {"kind": "completed", "value": "dispatched"},
            {"kind": "completed", "value": "final"},
        ]
    )
    adapter = CodexAppServerHarnessAdapter(fake)
    # The turn that originated a facade (captured server-side) completes cleanly here.
    first = adapter.start("a", capsule=None, outcomes=[])
    assert first.kind is HarnessResultKind.COMPLETION
    # The fabric re-dispatches with the child's settled outcome; it injects and resumes.
    done = adapter.start("a", capsule=first.capsule, outcomes=[_outcome("a:0")])
    assert done.kind is HarnessResultKind.COMPLETION and done.value == "final"
    assert fake.received_keys == ["idm-a:0"]
    assert fake.execution_count["a:0"] == 1
    assert fake.resumed == 1


def test_a_recommitted_outcome_is_not_reinjected() -> None:
    # A re-dispatch re-ships the same pending outcome against the advanced capsule; the
    # adapter dedupes injection by the committed fabric key, so it never injects twice.
    fake = FakeCodexAppServer(
        [
            {"kind": "completed", "value": "dispatched"},
            {"kind": "completed", "value": "more"},
            {"kind": "completed", "value": "final"},
        ]
    )
    adapter = CodexAppServerHarnessAdapter(fake)
    first = adapter.start("a", capsule=None, outcomes=[])
    outcome = _outcome("a:0")
    resumed = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
    # Re-ship the identical keyed outcome against the advanced capsule.
    adapter.start("a", capsule=resumed.capsule, outcomes=[outcome])
    assert fake.received_keys.count(outcome.idempotency_key) == 1
    assert fake.execution_count["a:0"] == 1


def test_crash_after_injection_before_terminal_does_not_reexecute() -> None:
    fake = FakeCodexAppServer(
        [
            {"kind": "completed", "value": "dispatched"},
            {"kind": "completed", "value": "final"},
        ]
    )
    first = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    outcome = _outcome("a:0")

    # The outcome injects and the turn completes, but the completion is lost to a crash,
    # so recovery re-dispatches from the pre-inject capsule with the same outcome.
    CodexAppServerHarnessAdapter(fake).start(
        "a", capsule=first.capsule, outcomes=[outcome]
    )
    assert fake.execution_count["a:0"] == 1

    adapter = CodexAppServerHarnessAdapter(fake)
    done = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
    # The stale capsule re-ships the inject, but the rollout dedupes it by key: the
    # effect ran exactly once and the episode still completes.
    assert done.kind is HarnessResultKind.COMPLETION and done.value == "final"
    assert fake.execution_count["a:0"] == 1


def test_codex_home_isolates_activations_but_is_stable() -> None:
    root = Path("/results")
    home = _isolated_codex_home(root, "wfl-1", "act-1")
    # Stable across steps of the same activation, so its rollout resumes.
    assert home == _isolated_codex_home(root, "wfl-1", "act-1")
    # Distinct per workflow and per task, so rollouts never co-mingle on disk.
    assert home != _isolated_codex_home(root, "wfl-2", "act-1")
    assert home != _isolated_codex_home(root, "wfl-1", "act-2")


def test_codex_home_sanitizes_path_separators() -> None:
    home = _isolated_codex_home(Path("/results"), "wfl-1", "op/../escape")
    assert home == Path("/results/codex_home/wfl-1/op_.._escape")
    assert Path("/results/codex_home") in home.parents


def _agent_spec(**fields: object) -> AgentSpecStrict:
    return AgentSpecStrict(taskType=TaskType.AGENT, **fields)  # type: ignore[arg-type]


def test_agent_task_reads_spec_task_then_data_task() -> None:
    assert _agent_task(_agent_spec(task="solve it")) == "solve it"
    assert _agent_task(_agent_spec(data={"task": "from data"})) == "from data"
    # spec.task wins over spec.data.task.
    assert _agent_task(_agent_spec(task="win", data={"task": "lose"})) == "win"


def test_agent_task_requires_a_task() -> None:
    with pytest.raises(ValueError, match="spec.task"):
        _agent_task(_agent_spec())

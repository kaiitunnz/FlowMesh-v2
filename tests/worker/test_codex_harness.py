"""The Codex app-server adapter maps the harness contract and recovers on the rollout.

A fake app-server scripts terminal turns so the binding runs without a live Codex: each
turn completes or fails, and a delivered outcome injects back and resumes the rollout.
A facade originates at the gateway, not the adapter, so the adapter never observes one.
The load-bearing part is crash recovery — across an app-server loss against the same
persisted rollout, a re-delivered outcome injects at most once, gated by the committed
fabric idempotency key rather than a Codex-local call id.
"""

import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pytest

from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    DeliveredOutcome,
    HarnessCapsule,
    HarnessResultKind,
    OutcomeKind,
)
from shared.tasks.specs import AgentSpecStrict
from shared.tasks.task_type import TaskType
from worker.executors.harness.codex import (
    CodexAppServerHarnessAdapter,
    CodexEvent,
    CodexInjectItem,
    CodexTurnCancelled,
    _agent_task,
)
from worker.utils import subreaper


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

    def cancel(self, thread_id: str | None) -> None:
        self.cancelled += 1

    def quiesce(self) -> bool:
        return True


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


def test_the_required_facades_are_mediated() -> None:
    adapter = CodexAppServerHarnessAdapter(FakeCodexAppServer([]))
    assert adapter.mediated_facades() >= REQUIRED_MEDIATED_FACADES


def test_a_completed_turn_completes_the_episode() -> None:
    fake = FakeCodexAppServer([{"kind": "completed", "value": "done"}])
    result = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.COMPLETION and result.value == "done"


def test_a_turn_error_fails_the_episode() -> None:
    fake = FakeCodexAppServer([{"kind": "error", "value": "boom"}])
    result = CodexAppServerHarnessAdapter(fake).start("a", capsule=None, outcomes=[])
    assert result.kind is HarnessResultKind.FAILURE and result.error == "boom"


def test_a_cancel_before_the_turn_starts_ends_the_step() -> None:
    fake = FakeCodexAppServer([{"kind": "completed", "value": "done"}])
    adapter = CodexAppServerHarnessAdapter(fake)
    adapter.cancel("a")
    with pytest.raises(CodexTurnCancelled):
        adapter.start("a", capsule=None, outcomes=[])
    assert fake.cursor == 0
    # The cancel was the step's; the next step runs.
    assert adapter.start("a", capsule=None, outcomes=[]).value == "done"


def test_a_cancel_after_the_step_ended_still_closes_the_app_server() -> None:
    fake = FakeCodexAppServer([{"kind": "completed", "value": "done"}])
    adapter = CodexAppServerHarnessAdapter(fake)
    assert adapter.start("a", capsule=None, outcomes=[]).value == "done"

    adapter.cancel("a")

    assert fake.cancelled == 1


@pytest.mark.parametrize("resume", [False, True])
def test_a_cancel_ends_a_step_still_opening_its_thread(resume: bool) -> None:
    opening = threading.Event()
    closed = threading.Event()

    class _HungAppServer(FakeCodexAppServer):
        def thread_start(self) -> str:
            opening.set()
            closed.wait(5)
            raise RuntimeError("the Codex app-server closed")

        def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
            self.thread_start()

        def cancel(self, thread_id: str | None) -> None:
            super().cancel(thread_id)
            closed.set()

    fake = _HungAppServer([{"kind": "completed", "value": "done"}])
    adapter = CodexAppServerHarnessAdapter(fake)
    capsule = (
        HarnessCapsule(
            backend=adapter.backend_key(),
            blob='{"thread_id": "thr-1", "rollout_ref": "thr-1"}',
        )
        if resume
        else None
    )

    def give_up() -> None:
        opening.wait(5)
        adapter.cancel("a")

    canceller = threading.Thread(target=give_up)
    canceller.start()

    with pytest.raises(RuntimeError, match="closed"):
        adapter.start("a", capsule=capsule, outcomes=[])
    canceller.join(5)

    assert fake.cancelled == 1
    assert fake.cursor == 0


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


def test_the_backend_binds_the_materialized_components_as_its_home_and_cwd(
    tmp_path: Path,
) -> None:
    """Codex reaches a home and a working directory only through private state."""
    pytest.importorskip("openai_codex")
    from worker.executors.harness.codex_transport import CodexTransportConfig

    home = tmp_path / "harness_home_fs"
    workspace = tmp_path / "workspace_fs"
    home.mkdir()
    workspace.mkdir()

    config = CodexTransportConfig(
        base_url="http://gw",
        model="m",
        codex_home=home,
        cwd=workspace,
        initial_input="t",
        task_id="tsk-1",
    ).to_codex_config()

    assert config.env is not None and config.env["CODEX_HOME"] == home.as_posix()
    assert config.cwd == workspace.as_posix()


def test_the_app_server_launches_under_its_supervisor_without_plugin_sync(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openai_codex")
    from worker.executors.harness.codex_transport import CodexTransportConfig

    config = CodexTransportConfig(
        base_url="http://gw",
        model="m",
        codex_home=tmp_path,
        initial_input="t",
        task_id="tsk-1",
    ).to_codex_config()

    launch = config.launch_args_override
    assert launch is not None
    assert launch[2] == subreaper.__file__
    command = launch[launch.index("--") + 1 :]
    assert command[-3:] == ("app-server", "--listen", "stdio://")
    assert "features.plugins=false" in command
    assert 'model_provider="flowmesh"' in command
    assert config.env is not None and config.env["PATH"]

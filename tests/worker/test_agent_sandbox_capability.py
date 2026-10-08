"""The capability gate over an agent's local commands, and the scripted exec step.

A command runs only under the capability the dispatch minted, which carries the fences
of the attachment that owns the workspace. Several commands run inside one dispatch: a
local action is not a boundary, so the episode neither yields nor reports between them.
"""

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from shared.harness import HarnessBackendKey, HarnessResultKind
from shared.private_state import PrivateStateAttachment
from shared.private_state.manifest import StateComponentKind
from shared.private_state.reference import BundleProfile
from shared.sandbox import (
    LocalSandboxCapability,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
    SandboxEgressMode,
    SandboxReapUnproved,
    SandboxRuntimeProfile,
)
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep
from worker.private_state import MaterializedState
from worker.sandbox import AgentSandboxRuntime
from worker.sandbox.runtime import SandboxRuntime

_ATTACHMENT = PrivateStateAttachment(
    attachment_id="psa-1",
    reference_id="aps-1",
    generation=3,
    worker_id="wkr-1",
    incarnation=7,
    write_epoch=4,
)


def _capability(**overrides: object) -> LocalSandboxCapability:
    fields: dict = {
        "attachment_id": "psa-1",
        "reference_id": "aps-1",
        "worker_id": "wkr-1",
        "incarnation": 7,
        "write_epoch": 4,
        "profile": SandboxRuntimeProfile(),
    }
    fields.update(overrides)
    return LocalSandboxCapability(**fields)


class _RecordingRuntime(SandboxRuntime):
    name = "recording"

    def __init__(self) -> None:
        self.commands: list[tuple[Path, tuple[str, ...]]] = []
        self.egress: list[bool] = []

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        self.commands.append((root, command.argv))
        self.egress.append(egress)
        return SandboxCommandResult(
            exit_code=0, stdout=f"ran {command.argv[0]}", stderr=""
        )


@pytest.fixture
def state(tmp_path: Path) -> MaterializedState:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return MaterializedState(
        reference_id="aps-1",
        generation=3,
        profile=BundleProfile.AGENT_HARNESS,
        components={StateComponentKind.WORKSPACE_FS: workspace},
    )


def test_a_command_runs_in_the_activations_own_workspace(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)

    result = sandbox.execute(SandboxCommand(argv=("echo", "hi")))

    assert result.exit_code == 0
    assert runtime.commands == [(state.workspace, ("echo", "hi"))]


def test_a_superseded_write_epoch_cannot_run_a_command(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(
        _capability(write_epoch=3), _ATTACHMENT, state, runtime
    )

    with pytest.raises(SandboxDenied):
        sandbox.execute(SandboxCommand(argv=("echo", "hi")))

    assert runtime.commands == []


def test_another_holders_capability_cannot_run_a_command(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(
        _capability(worker_id="wkr-2"), _ATTACHMENT, state, runtime
    )

    with pytest.raises(SandboxDenied):
        sandbox.execute(SandboxCommand(argv=("echo", "hi")))

    assert runtime.commands == []


def test_a_capability_for_another_lineage_cannot_run_a_command(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(
        _capability(reference_id="aps-2"), _ATTACHMENT, state, runtime
    )

    with pytest.raises(SandboxDenied):
        sandbox.execute(SandboxCommand(argv=("echo", "hi")))

    assert runtime.commands == []


def test_many_commands_run_inside_one_dispatch(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)
    adapter = ScriptedHarnessAdapter(
        [
            ScriptedStep(op="exec", call="c1", argv=["echo", "one"]),
            ScriptedStep(op="exec", call="c2", argv=["echo", "two"]),
            ScriptedStep(op="exec", call="c3", argv=["echo", "three"]),
            ScriptedStep(op="complete", value_from="c3"),
        ],
        "v1",
        sandbox,
    )

    result = adapter.start("act-1", capsule=None, outcomes=[])

    # One dispatch ran every command and completed: no boundary, no yield between them.
    assert result.kind is HarnessResultKind.COMPLETION
    assert result.value == "ran echo"
    assert len(runtime.commands) == 3


def test_an_agent_without_a_sandbox_cannot_run_a_command() -> None:
    adapter = ScriptedHarnessAdapter(
        [ScriptedStep(op="exec", call="c1", argv=["echo", "one"])], "v1"
    )

    with pytest.raises(SandboxDenied):
        adapter.start("act-1", capsule=None, outcomes=[])


def test_the_backend_key_is_unchanged_by_the_sandbox() -> None:
    adapter = ScriptedHarnessAdapter([], "v1")

    assert adapter.backend_key() == HarnessBackendKey(backend="scripted", version="v1")


def test_the_runtime_takes_its_fence_from_the_capability_not_the_command(state) -> None:
    """A command argument can never widen the fence the dispatch minted."""
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)

    sandbox.execute(SandboxCommand(argv=("curl", "https://example.com")))

    assert runtime.egress == [False]


def test_an_egress_minted_capability_relaxes_the_fence(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(
        _capability(network_egress=SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
        _ATTACHMENT,
        state,
        runtime,
    )

    sandbox.execute(SandboxCommand(argv=("curl", "https://example.com")))

    assert runtime.egress == [True]


class _UnprovedRuntime(_RecordingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.reaped = False

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        def retry() -> bool:
            self.reaped = True
            return True

        raise SandboxReapUnproved("a command left a process behind", retry=retry)


def test_a_closed_sandbox_admits_no_further_command(state) -> None:
    runtime = _RecordingRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)

    sandbox.close()

    with pytest.raises(SandboxDenied, match="ended"):
        sandbox.execute(SandboxCommand(argv=("ls",)))
    assert runtime.commands == []
    assert sandbox.drain()


def test_the_drain_waits_out_a_command_in_flight(state) -> None:
    started, release = threading.Event(), threading.Event()

    class _Slow(_RecordingRuntime):
        def run(self, *args: Any, **kwargs: Any) -> SandboxCommandResult:
            started.set()
            release.wait(5)
            return super().run(*args, **kwargs)

    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, _Slow())
    command = threading.Thread(
        target=sandbox.execute, args=(SandboxCommand(argv=("ls",)),), daemon=True
    )
    command.start()
    assert started.wait(5)
    drained: list[bool] = []
    drainer = threading.Thread(target=lambda: drained.append(sandbox.drain()))
    drainer.start()
    time.sleep(0.1)
    assert drained == []

    release.set()
    drainer.join(5)
    assert drained == [True]


def test_an_unproved_command_leaves_the_dispatch_unable_to_seal(state) -> None:
    runtime = _UnprovedRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)

    with pytest.raises(SandboxReapUnproved):
        sandbox.execute(SandboxCommand(argv=("make",)))
    assert sandbox.reap_unproved
    with pytest.raises(SandboxReapUnproved):
        sandbox.execute(SandboxCommand(argv=("ls",)))

    # The leftover tree is reaped on the drain, but the dispatch stays unsealable.
    assert not sandbox.drain()
    assert runtime.reaped
    assert sandbox.finish_reaps()


class _FlakyRetryRuntime(_RecordingRuntime):
    """A command left unproved whose first reap retry raises."""

    def __init__(self) -> None:
        super().__init__()
        self.retries = 0
        self.abandoned = 0

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        def retry() -> bool:
            self.retries += 1
            if self.retries == 1:
                raise OSError("the supervisor's pipe is gone")
            return True

        def abandon() -> None:
            self.abandoned += 1

        raise SandboxReapUnproved("left behind", retry=retry, abandon=abandon)


def test_a_reap_whose_retry_raises_is_kept_for_the_next_retry(state) -> None:
    runtime = _FlakyRetryRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)
    with pytest.raises(SandboxReapUnproved):
        sandbox.execute(SandboxCommand(argv=("make",)))

    assert not sandbox.finish_reaps()
    assert sandbox.finish_reaps()
    assert runtime.retries == 2


def test_abandoning_releases_every_unproved_reap(state) -> None:
    runtime = _FlakyRetryRuntime()
    sandbox = AgentSandboxRuntime(_capability(), _ATTACHMENT, state, runtime)
    with pytest.raises(SandboxReapUnproved):
        sandbox.execute(SandboxCommand(argv=("make",)))

    sandbox.abandon_reaps()

    assert runtime.abandoned == 1
    assert sandbox.finish_reaps()

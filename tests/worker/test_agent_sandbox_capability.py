"""The capability gate over an agent's local commands, and the scripted exec step.

A command runs only under the capability the dispatch minted, which carries the fences
of the attachment that owns the workspace. Several commands run inside one dispatch: a
local action is not a boundary, so the episode neither yields nor reports between them.
"""

from pathlib import Path

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

    def run(
        self, root: Path, command: SandboxCommand, profile: SandboxRuntimeProfile
    ) -> SandboxCommandResult:
        self.commands.append((root, command.argv))
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

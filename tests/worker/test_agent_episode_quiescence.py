"""A step seals only after every writer it bound to its attachment is proved stopped."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessQuiescenceError,
    HarnessResult,
    HarnessResultKind,
    MediatedFacade,
)
from shared.private_state import (
    ActivationPrivateStateReference,
    PrivateStateAttachment,
    PrivateStateBinding,
)
from shared.sandbox import (
    LocalSandboxCapability,
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
    SandboxReapUnproved,
    SandboxRuntimeProfile,
)
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.ids import new_private_state_reference_id
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import agent_episode_executor as aee
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.harness import register_adapter
from worker.private_state import PrivateStateHolder
from worker.sandbox import SandboxRuntime

_BACKEND = "fake-quiesce"


class _Adapter(HarnessAdapter):
    """Records the order of its own calls into a shared log beside the seal's."""

    def __init__(
        self,
        log: list[str],
        *,
        raises: BaseException | None = None,
        proves: bool = True,
        bypass: bool = False,
        write_to: Path | None = None,
    ) -> None:
        self._log = log
        self._raises = raises
        self._proves = proves
        self._bypass = bypass
        self._write_to = write_to

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend=_BACKEND, version="v1")

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        return frozenset() if self._bypass else REQUIRED_MEDIATED_FACADES

    def start(self, activation_id: str, *, capsule: Any, outcomes: Any) -> Any:
        self._log.append("start")
        if self._write_to is not None:
            (self._write_to / "rollout.jsonl").write_text("turn")
        if self._raises is not None:
            raise self._raises
        return HarnessResult(kind=HarnessResultKind.COMPLETION, value="done")

    def cancel(self, activation_id: str) -> None:
        self._log.append("cancel")

    def quiesce(self, activation_id: str) -> None:
        self._log.append("quiesce")
        if not self._proves:
            raise HarnessQuiescenceError("a writer outlived the step")


def _episode(
    tmp_path: Path, sandboxed: bool = False
) -> tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path]:
    binding = PrivateStateBinding(
        reference=ActivationPrivateStateReference(
            reference_id=new_private_state_reference_id(),
            instance_id="wfl-1",
            activation_id="act-1",
        )
    )
    attachment = PrivateStateAttachment(
        attachment_id="psa-1",
        reference_id=binding.reference.reference_id,
        generation=binding.generation,
        worker_id="wkr-1",
        incarnation=1,
        write_epoch=1,
    )
    sandbox = LocalSandboxCapability(
        attachment_id=attachment.attachment_id,
        reference_id=attachment.reference_id,
        worker_id=attachment.worker_id,
        incarnation=attachment.incarnation,
        write_epoch=attachment.write_epoch,
        profile=SandboxRuntimeProfile(),
    )
    message = make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={
            "backend": {"backend": _BACKEND, "version": "v1"},
            "private_state": binding.model_dump(mode="json"),
            "private_state_attachment": attachment.model_dump(mode="json"),
            "sandbox": sandbox.model_dump(mode="json") if sandboxed else None,
        },
    )
    root = tmp_path / "private"
    executor = AgentEpisodeExecutor(make_worker_config(private_state_dir=root))
    return executor, message, root / binding.reference.reference_id


@pytest.fixture
def episode(tmp_path: Path) -> tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path]:
    return _episode(tmp_path)


def _use(adapter: _Adapter) -> None:
    register_adapter(_BACKEND, lambda *_: adapter)


def _record_seals(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    seal = PrivateStateHolder.seal

    def recording(self: PrivateStateHolder, *args: Any) -> Any:
        log.append("seal")
        return seal(self, *args)

    monkeypatch.setattr(PrivateStateHolder, "seal", recording)


def test_the_seal_follows_the_harness_quiescing(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    _use(_Adapter(log))
    _record_seals(monkeypatch, log)

    result = executor.run(message, tmp_path)

    assert log == ["start", "quiesce", "seal"]
    assert result.private_state is not None


def test_an_unproved_quiescence_seals_nothing_and_fails_without_a_retry(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    _use(_Adapter(log, proves=False))
    _record_seals(monkeypatch, log)

    with pytest.raises(ExecutionError) as raised:
        executor.run(message, tmp_path)

    assert "PrivateStateUnavailable: quiescence_unproved" in str(raised.value)
    assert raised.value.retryable is False
    assert "seal" not in log
    # The lineage is refused from then on, even before any generation was sealed.
    with pytest.raises(ExecutionError, match="quiescence_unproved"):
        executor.run(message, tmp_path)


def test_a_raised_step_still_quiesces_its_harness(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    _use(_Adapter(log, raises=RuntimeError("the turn broke")))

    with pytest.raises(RuntimeError, match="the turn broke"):
        executor.run(message, tmp_path)

    assert log == ["start", "cancel", "quiesce"]


def test_a_step_refused_before_its_turn_still_quiesces_its_harness(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    _use(_Adapter(log, bypass=True))

    with pytest.raises(ExecutionError, match="does not mediate"):
        executor.run(message, tmp_path)

    assert log == ["cancel", "quiesce"]


def test_a_raised_step_whose_harness_will_not_quiesce_fails_unproved(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode
    _use(_Adapter([], raises=RuntimeError("the turn broke"), proves=False))

    with pytest.raises(ExecutionError) as raised:
        executor.run(message, tmp_path)

    assert "quiescence_unproved" in str(raised.value)
    assert raised.value.retryable is False


def test_a_cancelled_step_stays_cancelled_when_its_harness_will_not_quiesce(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode

    class _CancelledMidTurn(_Adapter):
        def start(self, activation_id: str, *, capsule: Any, outcomes: Any) -> Any:
            executor.cancel(message.task_id)
            raise RuntimeError("the app-server went away")

    _use(_CancelledMidTurn([], proves=False))

    with pytest.raises(TaskCancelledError):
        executor.run(message, tmp_path)


def test_a_later_cleanup_retries_an_unproved_teardown_until_it_is_proved(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    adapter = _Adapter(log, proves=False)
    _use(adapter)
    with pytest.raises(ExecutionError):
        executor.run(message, tmp_path)

    executor.cleanup_after_run()
    adapter._proves = True
    executor.cleanup_after_run()
    executor.cleanup_after_run()

    assert log.count("quiesce") == 3


def test_a_teardown_that_never_proves_is_given_up_after_its_attempts(
    episode: tuple[AgentEpisodeExecutor, WorkerTaskMessage, Path], tmp_path: Path
) -> None:
    executor, message, _ = episode
    log: list[str] = []
    _use(_Adapter(log, proves=False))
    with pytest.raises(ExecutionError):
        executor.run(message, tmp_path)

    for _ in range(5):
        executor.cleanup_after_run()

    assert log.count("quiesce") == 1 + aee._UNENDED_ATTEMPTS


def test_a_cancel_closes_the_sandbox_to_later_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _RecordingRuntime()
    monkeypatch.setattr(aee, "build_sandbox_runtime", lambda: runtime)
    executor, message, _ = _episode(tmp_path, sandboxed=True)
    denied: list[str] = []

    class _CancelledBetweenCommands(_SandboxedAdapter):
        def start(self, activation_id: str, *, capsule: Any, outcomes: Any) -> Any:
            self.sandbox.execute(SandboxCommand(argv=("first",)))
            executor.cancel(message.task_id)
            try:
                self.sandbox.execute(SandboxCommand(argv=("second",)))
            except SandboxDenied as exc:
                denied.append(str(exc))
            raise RuntimeError("the turn was given up")

    register_adapter(
        _BACKEND,
        lambda backend, task, config, facade, state, sandbox: (
            _CancelledBetweenCommands(sandbox)
        ),
    )

    with pytest.raises(TaskCancelledError):
        executor.run(message, tmp_path)

    assert runtime.commands == [("first",)]
    assert denied


def test_a_raised_step_releases_its_episode_once(tmp_path: Path) -> None:
    executor, message, _ = _episode(tmp_path)
    facade = MagicMock()
    executor._lifecycle = MagicMock(responses_facade=facade)
    _use(_Adapter([], raises=RuntimeError("the turn broke")))

    with pytest.raises(RuntimeError):
        executor.run(message, tmp_path)

    facade.refuse_episode.assert_called_once_with(message.task_id)
    facade.release_episode.assert_called_once_with(message.task_id)
    facade.unregister_episode.assert_called_once_with(message.task_id)


class _RecordingRuntime(SandboxRuntime):
    name = "recording"

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        self.commands.append(command.argv)
        return SandboxCommandResult(exit_code=0, stdout="", stderr="")


class _UnprovedRuntime(_RecordingRuntime):
    name = "unproved"

    def run(
        self,
        root: Path,
        command: SandboxCommand,
        profile: SandboxRuntimeProfile,
        egress: bool = False,
    ) -> SandboxCommandResult:
        raise SandboxReapUnproved("a command left a process behind")


class _SandboxedAdapter(_Adapter):
    def __init__(self, sandbox: LocalSandboxExecutor | None) -> None:
        super().__init__([])
        assert sandbox is not None
        self.sandbox = sandbox

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        return REQUIRED_MEDIATED_FACADES | {MediatedFacade.SANDBOX}


def test_an_unproved_command_fails_the_step_even_when_the_harness_caught_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aee, "build_sandbox_runtime", _UnprovedRuntime)
    executor, message, _ = _episode(tmp_path, sandboxed=True)

    class _Catching(_SandboxedAdapter):
        def start(self, activation_id: str, *, capsule: Any, outcomes: Any) -> Any:
            try:
                self.sandbox.execute(SandboxCommand(argv=("make",)))
            except SandboxReapUnproved:
                pass
            return HarnessResult(kind=HarnessResultKind.COMPLETION, value="done")

    register_adapter(
        _BACKEND,
        lambda backend, task, config, facade, state, sandbox: _Catching(sandbox),
    )

    with pytest.raises(ExecutionError, match="quiescence_unproved"):
        executor.run(message, tmp_path)

"""A sandbox session runs one declared command per step against its private tree.

The bound generation indexes the command a step runs, each step seals the next
generation, the final command completes the session with every command's result, and a
generation the holder cannot supply fails the step closed.
"""

import json
from pathlib import Path

import pytest

from shared.harness import HarnessResultKind, SandboxSessionDispatch
from shared.private_state import (
    ActivationPrivateStateReference,
    BundleProfile,
    OwnerFence,
    PrivateStateAttachment,
    PrivateStateBinding,
    StateBundleManifest,
)
from shared.schemas.result import SandboxResult
from shared.tasks.specs import SandboxSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors.base_executor import ExecutionError
from worker.executors.sandbox_session_executor import SandboxSessionExecutor

_REFERENCE = "aps-session"


def _spec(*commands: list[str]) -> SandboxSpecStrict:
    return SandboxSpecStrict.model_validate(
        {
            "taskType": "sandbox",
            "commands": [{"argv": argv} for argv in commands],
        }
    )


def _binding(
    generation: int = 0,
    manifest: StateBundleManifest | None = None,
) -> PrivateStateBinding:
    reference = ActivationPrivateStateReference(
        reference_id=_REFERENCE,
        instance_id="wfl-1",
        activation_id="act-1",
        profile=BundleProfile.SANDBOX_SESSION,
    )
    owner = OwnerFence(worker_id="wrk-1", incarnation=1) if generation else None
    return PrivateStateBinding(
        reference=reference,
        generation=generation,
        manifest=manifest,
        owner=owner,
    )


def _attachment(
    binding: PrivateStateBinding, write_epoch: int
) -> PrivateStateAttachment:
    return PrivateStateAttachment(
        attachment_id=f"psa-{write_epoch}",
        reference_id=_REFERENCE,
        generation=binding.generation,
        worker_id="wrk-1",
        incarnation=1,
        write_epoch=write_epoch,
    )


def _step(executor, spec, binding, epoch, index, capsule=None):
    task = make_worker_task_message(
        spec,
        task_type=TaskType.SANDBOX,
        sandbox_session=SandboxSessionDispatch(
            command_index=index,
            capsule_blob=capsule,
            private_state=binding,
            private_state_attachment=_attachment(binding, epoch),
        ),
    )
    return executor.run(task, Path("."))


@pytest.fixture
def executor(tmp_path: Path) -> SandboxSessionExecutor:
    return SandboxSessionExecutor(
        make_worker_config(private_state_dir=tmp_path / "private")
    )


def test_a_single_command_session_completes_with_its_result(executor) -> None:
    spec = _spec(["sh", "-c", "echo one"])

    out = _step(executor, spec, _binding(), 1, 0)

    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value is not None
    result = SandboxResult.model_validate_json(out.value)
    assert [item.argv for item in result.commands] == [["sh", "-c", "echo one"]]
    assert result.commands[0].stdout.strip() == "one"
    assert out.private_state is not None
    assert out.private_state.manifest.generation == 1


def test_a_later_command_resumes_the_sealed_tree_and_accrues_history(executor) -> None:
    spec = _spec(["sh", "-c", "echo first > note.txt"], ["cat", "note.txt"])

    first = _step(executor, spec, _binding(), 1, 0)
    assert first.harness_result.kind is HarnessResultKind.YIELD
    manifest = first.private_state.manifest
    capsule = first.harness_result.capsule.blob

    second = _step(executor, spec, _binding(1, manifest), 2, 1, capsule)

    assert second.harness_result.kind is HarnessResultKind.COMPLETION
    assert second.value is not None
    result = SandboxResult.model_validate_json(second.value)
    # The second command reads what the first wrote, so the session accrued real state.
    assert result.commands[1].stdout.strip() == "first"
    assert len(result.commands) == 2
    assert second.private_state.manifest.generation == 2


def test_two_sessions_on_one_holder_do_not_see_each_other(tmp_path: Path) -> None:
    root = tmp_path / "private"
    spec = _spec(["sh", "-c", "ls | wc -l"])
    for reference in ("aps-a", "aps-b"):
        executor = SandboxSessionExecutor(make_worker_config(private_state_dir=root))
        binding = PrivateStateBinding(
            reference=ActivationPrivateStateReference(
                reference_id=reference,
                instance_id="wfl-1",
                activation_id=f"act-{reference}",
                profile=BundleProfile.SANDBOX_SESSION,
            )
        )
        task = make_worker_task_message(
            spec,
            task_type=TaskType.SANDBOX,
            sandbox_session=SandboxSessionDispatch(
                command_index=0,
                private_state=binding,
                private_state_attachment=PrivateStateAttachment(
                    attachment_id="psa-1",
                    reference_id=reference,
                    generation=0,
                    worker_id="wrk-1",
                    incarnation=1,
                    write_epoch=1,
                ),
            ),
        )
        out = executor.run(task, Path("."))
        assert out.value is not None
        result = SandboxResult.model_validate_json(out.value)
        # Each session starts in its own empty tree, so neither sees the other.
        assert result.commands[0].stdout.strip() == "0"


def test_an_unsupplied_generation_fails_the_step_closed(executor, tmp_path) -> None:
    spec = _spec(["sh", "-c", "echo one > note.txt"], ["cat", "note.txt"])
    first = _step(executor, spec, _binding(), 1, 0)
    manifest = first.private_state.manifest
    # The sealed tree is edited behind the session's back, so the bound generation can
    # no longer be supplied in full.
    (tmp_path / "private" / _REFERENCE / "sandbox_fs" / "note.txt").write_text("other")

    with pytest.raises(ExecutionError, match="PrivateStateUnavailable"):
        _step(
            executor,
            spec,
            _binding(1, manifest),
            2,
            1,
            first.harness_result.capsule.blob,
        )


def test_a_step_outside_the_declared_commands_is_refused(executor) -> None:
    with pytest.raises(ExecutionError, match="outside its declared"):
        _step(executor, _spec(["true"]), _binding(), 1, 5)


def test_an_unreadable_continuation_is_refused(executor) -> None:
    spec = _spec(["true"], ["true"])
    with pytest.raises(ExecutionError, match="continuation is unreadable"):
        _step(executor, spec, _binding(), 1, 0, capsule="{not json")


def test_a_session_without_state_authority_is_refused(executor) -> None:
    task = make_worker_task_message(
        _spec(["true"]),
        task_type=TaskType.SANDBOX,
        sandbox_session=SandboxSessionDispatch(command_index=0),
    )
    with pytest.raises(ExecutionError, match="no private-state authority"):
        executor.run(task, Path("."))


def test_history_round_trips_through_the_capsule(executor) -> None:
    spec = _spec(["sh", "-c", "echo a"], ["sh", "-c", "echo b"])
    first = _step(executor, spec, _binding(), 1, 0)
    recorded = json.loads(first.harness_result.capsule.blob)
    assert [entry["argv"] for entry in recorded] == [["sh", "-c", "echo a"]]

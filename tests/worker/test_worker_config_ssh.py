"""A worker reads its SSH session settings at its config edge."""

from pathlib import Path

import pytest

from shared.tasks.specs import SSHSpecStrict
from worker.config import WorkerConfig
from worker.executors.ssh_session import SSHConfig


@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("WORKER_TOKEN", "tok")
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "127.0.0.1:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.delenv("ENABLE_SSH_GPU_LIMIT", raising=False)
    return monkeypatch


def test_a_session_gets_only_the_gpus_its_spec_requests_by_default(
    worker_env: pytest.MonkeyPatch,
) -> None:
    worker_env.setenv("WORKER_HOST_GPU_ID", "0,1,2,3")
    spec = SSHSpecStrict.model_validate(
        {
            "taskType": "ssh",
            "command": ["nvidia-smi"],
            "resources": {"hardware": {"gpu": {"count": 1}}},
        }
    )

    cfg = SSHConfig.from_spec(spec, WorkerConfig.from_env())

    assert cfg.gpu_device_ids == ["0"]


def test_the_gpu_limit_can_be_turned_off(worker_env: pytest.MonkeyPatch) -> None:
    worker_env.setenv("ENABLE_SSH_GPU_LIMIT", "false")

    assert WorkerConfig.from_env().enable_ssh_gpu_limit is False


def test_a_session_and_teardown_stop_within_the_configured_timeout(
    worker_env: pytest.MonkeyPatch,
) -> None:
    worker_env.setenv("SSH_STOP_TIMEOUT_SEC", "7.5")
    spec = SSHSpecStrict.model_validate({"taskType": "ssh", "command": ["true"]})

    config = WorkerConfig.from_env()

    assert config.ssh_stop_timeout_sec == 7.5
    assert SSHConfig.from_spec(spec, config).stop_timeout_sec == 7.5


def test_a_session_config_keeps_what_the_spec_itself_named(
    worker_env: pytest.MonkeyPatch,
) -> None:
    config = WorkerConfig.from_env()
    bare = SSHSpecStrict.model_validate({"taskType": "ssh", "command": ["true"]})
    named = SSHSpecStrict.model_validate(
        {"taskType": "ssh", "command": ["true"], "image": "img", "user": "alice"}
    )

    defaulted = SSHConfig.from_spec(bare, config)
    chosen = SSHConfig.from_spec(named, config)

    assert defaulted.image and defaulted.user
    assert (defaulted.requested_image, defaulted.requested_user) == (None, None)
    assert (chosen.requested_image, chosen.requested_user) == ("img", "alice")

"""Tests for server worker configuration models."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from server.hooks import PrincipalContext
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
)
from server.supervisor.adapters.ssh import SSHConfig
from server.supervisor.adapters.vastai import VastAIWorkerAdapter, VastAIWorkerConfig
from server.supervisor.manager import ServerWorkerConfig, WorkerInitConfig
from server.utils.helpers import ResourcePool
from shared.schemas.worker import SSHBackendName


class TestWorkerInitConfig:
    def test_defaults(self) -> None:
        cfg = WorkerInitConfig()
        assert cfg.provider == "docker"
        assert cfg.init_on_start is True
        assert cfg.worker_config == {}

    def test_custom_provider(self) -> None:
        cfg = WorkerInitConfig(provider="vastai", init_on_start=False)
        assert cfg.provider == "vastai"
        assert cfg.init_on_start is False

    def test_extra_fields_preserved(self) -> None:
        cfg = WorkerInitConfig(  # type: ignore[call-arg]
            provider="docker",
            worker_config={"image": "my-image:latest"},
            custom_field="custom_value",
        )
        extras = cfg.extra_kwargs
        assert "custom_field" in extras
        assert extras["custom_field"] == "custom_value"
        assert "provider" not in extras
        assert "worker_config" not in extras

    def test_worker_config_nested(self) -> None:
        cfg = WorkerInitConfig(
            worker_config={
                "image": "flowmesh_worker:gpu",
                "gpu_count": 4,
                "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            }
        )
        assert cfg.worker_config["gpu_count"] == 4


class TestServerWorkerConfig:
    def test_defaults(self) -> None:
        cfg = ServerWorkerConfig()
        assert cfg.default_worker_config == {}
        assert cfg.workers == []

    def test_with_workers(self) -> None:
        cfg = ServerWorkerConfig(
            default_worker_config={"tags": "gpu"},
            workers=[
                WorkerInitConfig(provider="docker"),
                WorkerInitConfig(provider="vastai", init_on_start=False),
            ],
        )
        assert len(cfg.workers) == 2
        assert cfg.workers[0].provider == "docker"
        assert cfg.workers[1].provider == "vastai"


class TestVastAISessionBackend:
    """A VastAI instance is the worker container and exposes no Docker socket."""

    def test_docker_backend_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be 'docker'"):
            VastAIWorkerConfig(ssh=SSHConfig(session_backend=SSHBackendName.DOCKER))

    def test_rejection_ignores_case_and_padding(self) -> None:
        with pytest.raises(ValidationError, match="cannot be 'docker'"):
            VastAIWorkerConfig(
                ssh=SSHConfig.model_validate({"session_backend": "  Docker  "})
            )

    @pytest.mark.parametrize(
        "backend", [SSHBackendName.PROCESS, SSHBackendName.AUTO, None]
    )
    def test_other_backends_are_accepted(self, backend: SSHBackendName | None) -> None:
        cfg = VastAIWorkerConfig(ssh=SSHConfig(session_backend=backend))
        assert cfg.ssh.session_backend is backend


class TestSSHEnvironment:
    def test_the_session_backend_and_direct_host_reach_the_worker(self) -> None:
        env = SSHConfig(
            session_backend=SSHBackendName.PROCESS, direct_host="100.64.0.7"
        ).to_env(True)

        assert env["SSH_SESSION_BACKEND"] == "process"
        assert env["SSH_DIRECT_HOST"] == "100.64.0.7"
        assert env["ENABLE_SSH_GPU_LIMIT"] == "1"

    def test_a_vastai_worker_with_ssh_gets_the_ssh_environment(self) -> None:
        adapter = _vastai_adapter(
            VastAIWorkerConfig(
                enable_ssh=True, ssh=SSHConfig(session_backend=SSHBackendName.PROCESS)
            )
        )

        assert adapter._base_environment()["SSH_SESSION_BACKEND"] == "process"

    def test_a_direct_host_reaches_only_the_worker_it_is_set_for(self) -> None:
        assert "SSH_DIRECT_HOST" not in SSHConfig().to_env(True)

    def test_an_enabled_worker_with_no_backend_named_gets_auto(self) -> None:
        env = SSHConfig(session_backend=None).to_env(True)

        assert env["SSH_SESSION_BACKEND"] == "auto"

    @pytest.mark.parametrize(
        "backend", [None, SSHBackendName.AUTO, SSHBackendName.PROCESS]
    )
    def test_a_worker_without_ssh_is_told_to_serve_none(
        self, backend: SSHBackendName | None
    ) -> None:
        adapter = _vastai_adapter(
            VastAIWorkerConfig(
                enable_ssh=False,
                ssh=SSHConfig(session_backend=backend, default_ttl_sec=60),
            )
        )

        environment = adapter._base_environment()

        assert environment["SSH_SESSION_BACKEND"] == "off"
        assert "SSH_DEFAULT_TTL_SEC" not in environment

    def test_a_stack_wide_docker_backend_is_refused_on_vastai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The stack-wide SSH_SESSION_BACKEND reaches a VastAI worker as the default.
        monkeypatch.setattr(
            VastAIWorkerConfig.model_fields["ssh"],
            "default_factory",
            lambda: SSHConfig(session_backend=SSHBackendName.DOCKER),
        )
        VastAIWorkerConfig.model_rebuild(force=True)
        try:
            with pytest.raises(ValidationError, match="cannot be 'docker'"):
                VastAIWorkerConfig()
        finally:
            monkeypatch.undo()
            VastAIWorkerConfig.model_rebuild(force=True)


@pytest.mark.parametrize(
    ("backend", "mounted"),
    [
        (None, True),
        (SSHBackendName.AUTO, True),
        (SSHBackendName.DOCKER, True),
        (SSHBackendName.PROCESS, False),
    ],
)
def test_a_docker_worker_gets_the_socket_only_for_docker_sessions(
    backend: SSHBackendName | None, mounted: bool
) -> None:
    adapter = object.__new__(DockerWorkerAdapter)
    adapter.config = DockerWorkerConfig(
        worker_type=WorkerType.CPU,
        enable_ssh=True,
        ssh=SSHConfig(session_backend=backend),
    )
    volumes: list[str] = []

    adapter._mount_docker_socket(volumes)

    assert bool(volumes) is mounted


def _vastai_adapter(config: VastAIWorkerConfig) -> VastAIWorkerAdapter:
    return VastAIWorkerAdapter(
        token=WorkerTokenType("vast_0.token"),
        alias="vast_0",
        config=config,
        vastai_client=MagicMock(),
        instance_pool=ResourcePool(),
        owner=PrincipalContext(
            principal_id="u",
            org_id="o",
            external_id="u",
            principal_type="user",
            scopes=[],
        ),
    )

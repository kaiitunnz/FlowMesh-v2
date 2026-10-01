"""Tests for how the worker manager builds its providers."""

import logging
from unittest.mock import MagicMock

import pytest
from docker.errors import DockerException

from server.hooks import PrincipalContext
from server.supervisor import manager as manager_module
from server.supervisor.manager import (
    ProviderUnavailableError,
    WorkerInitConfig,
    WorkerManager,
)
from server.supervisor.registry import WorkerRegistry


def _manager() -> WorkerManager:
    return WorkerManager(
        system_principal=MagicMock(spec=PrincipalContext),
        config_path="unused",
        registry=WorkerRegistry(),
        logger=logging.getLogger("test.supervisor"),
    )


def test_an_unavailable_docker_provider_leaves_the_others(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_docker(_: PrincipalContext) -> None:
        raise DockerException("Error while fetching server API version")

    vastai_spec = MagicMock()
    vastai_spec.name = "vastai"
    monkeypatch.setattr(manager_module, "docker_provider_spec", no_docker)
    monkeypatch.setattr(manager_module, "vastai_provider_spec", lambda _: vastai_spec)

    with caplog.at_level(logging.WARNING, logger="test.supervisor"):
        wm = _manager()

    assert wm._providers.keys() == {"external", "vastai"}
    assert wm._providers["vastai"] is vastai_spec
    assert any(
        "Docker worker provider unavailable" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_a_worker_of_an_unavailable_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_: PrincipalContext) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(manager_module, "docker_provider_spec", unavailable)
    monkeypatch.setattr(manager_module, "vastai_provider_spec", unavailable)
    wm = _manager()
    wm._is_started = True

    with pytest.raises(ProviderUnavailableError, match="'docker' is not available"):
        await wm.create_worker(WorkerInitConfig(provider="docker"))

"""Tests for `worker.executors.utils.artifacts`.

Pure-logic helpers (no pandas/datasets/PIL) — runnable in CI without
the worker-side extras.
"""

from pathlib import Path
from typing import Any

import pytest

from shared.schemas.artifact import ArtifactContext
from shared.schemas.result import BaseExecutorResult
from shared.tasks import TaskType
from shared.tasks.specs import EchoSpecStrict
from worker.executors.base_executor import ExecutionError
from worker.executors.utils import artifacts as artifacts_module
from worker.executors.utils.artifacts import (
    artifact_ref_source,
    artifact_ref_text,
    hydrated_artifact_ref,
    is_flowmesh_origin_url,
    resolve_artifact,
)


def _upstream() -> dict[str, BaseExecutorResult]:
    return {
        "producer": BaseExecutorResult(_artifacts=ArtifactContext(task_id="tsk-prod"))
    }


class TestArtifactRefSource:
    def test_names_the_producer_and_path(self) -> None:
        assert artifact_ref_source({"path": "a.png"}, _upstream(), "producer") == (
            "tsk-prod",
            "a.png",
        )

    def test_a_pre_upgrade_context_names_no_producer(self) -> None:
        upstream = BaseExecutorResult.model_validate(
            {"_artifacts": {"base_dir": "/results/tsk-old", "base_url": "http://x"}}
        )
        with pytest.raises(ExecutionError, match="_artifacts context is missing"):
            artifact_ref_source({"path": "a.png"}, {"producer": upstream}, "producer")

    def test_missing_context_raises(self) -> None:
        with pytest.raises(ExecutionError, match="_artifacts context is missing"):
            artifact_ref_source(
                {"path": "a.png"}, {"producer": BaseExecutorResult()}, "producer"
            )

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ExecutionError, match="non-empty 'path' field"):
            artifact_ref_source({"path": ""}, _upstream(), "producer")

    def test_other_values_name_nothing(self) -> None:
        for value in ("hello", 42, [1, 2], {"url": "http://x"}):
            assert artifact_ref_source(value, None, None) is None


class TestArtifactRefs:
    def test_a_hydrated_ref_is_its_local_path(self) -> None:
        spec = EchoSpecStrict(taskType=TaskType.ECHO)
        spec.record_hydrated_artifacts({("tsk-prod", "a.png"): "/in/a.png"})

        assert (
            hydrated_artifact_ref({"path": "a.png"}, spec, _upstream(), "producer")
            == "/in/a.png"
        )
        assert hydrated_artifact_ref("plain", spec, None, None) == "plain"

    def test_a_ref_nothing_hydrated_is_unreadable(self) -> None:
        spec = EchoSpecStrict(taskType=TaskType.ECHO)

        with pytest.raises(ExecutionError, match="input_unreadable.*no artifact"):
            hydrated_artifact_ref({"path": "a.png"}, spec, _upstream(), "producer")

    def test_a_ref_as_text_names_its_published_artifact(self) -> None:
        assert (
            artifact_ref_text({"path": "a.png"}, _upstream(), "producer")
            == "/api/v1/results/tsk-prod/files/a.png"
        )


class TestResolveArtifact:
    def test_local_path_still_symlinks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail_if_called(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("requests.get must not be called for a local path")

        monkeypatch.setattr(artifacts_module.requests, "get", _fail_if_called)

        src = tmp_path / "a.png"
        src.write_bytes(b"\x89PNG")

        local_path = resolve_artifact(src.as_posix())

        assert local_path.is_symlink()
        assert local_path.resolve() == src.resolve()

    def test_external_url_sends_no_auth_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

        captured: dict[str, Any] = {}

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"external-bytes"]

        def _fake_get(
            url: str, headers: dict[str, str] | None, stream: bool, timeout: float
        ) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(artifacts_module.requests, "get", _fake_get)

        resolve_artifact("https://attacker.example/x.png")

        assert captured["url"] == "https://attacker.example/x.png"
        assert captured["headers"] is None

    def test_origin_url_sends_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOWMESH_API_KEY", "s3cr3t-token")
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://worker-b.internal")

        captured: dict[str, Any] = {}

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"origin-bytes"]

        def _fake_get(
            url: str, headers: dict[str, str] | None, stream: bool, timeout: float
        ) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(artifacts_module.requests, "get", _fake_get)

        resolve_artifact("https://worker-b.internal/api/v1/results/tsk-1/files/a.png")

        assert captured["headers"] is not None
        assert captured["headers"].get("Authorization") == "Bearer s3cr3t-token"


class TestIsFlowmeshOriginUrl:
    def test_default_port_equivalence_target_implicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh:443")
        assert is_flowmesh_origin_url("https://flowmesh/api/v1/results") is True

    def test_default_port_equivalence_base_implicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh")
        assert is_flowmesh_origin_url("https://flowmesh:443/api/v1/results") is True

    def test_different_host_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh")
        assert is_flowmesh_origin_url("https://attacker.example/x") is False

    def test_different_explicit_non_default_port_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh:8443")
        assert is_flowmesh_origin_url("https://flowmesh:9443/x") is False

    def test_empty_base_url_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLOWMESH_BASE_URL", raising=False)
        assert is_flowmesh_origin_url("https://flowmesh/x") is False

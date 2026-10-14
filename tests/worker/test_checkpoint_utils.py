"""Tests for the worker artifact helpers in checkpoints."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from worker.executors.base_executor import TaskReference
from worker.executors.utils import checkpoints


def _task() -> TaskReference:
    return cast(TaskReference, SimpleNamespace(task_id="task-1", spec=None))


def _traces(out_dir: Path) -> None:
    (out_dir / "logs").mkdir(parents=True)
    (out_dir / "logs" / "spans.jsonl").write_text("{}\n", encoding="utf-8")
    (out_dir / "logs" / "lineage.jsonl").write_text("{}\n", encoding="utf-8")


class TestMaybeUploadTraces:
    def test_posts_each_trace_to_the_server_with_its_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
        _traces(tmp_path)
        posted: list[tuple[str, str | None]] = []

        class _Response:
            def raise_for_status(self) -> None:
                return None

        def post(url: str, files: Any, headers: dict[str, str], timeout: float) -> Any:
            posted.append((url, headers.get("Authorization")))
            return _Response()

        monkeypatch.setattr(checkpoints.requests, "post", post)

        assert checkpoints.maybe_upload_traces(
            _task(), tmp_path, "http://root:8000/"
        ) == [
            "spans",
            "lineage",
        ]
        assert posted == [
            ("http://root:8000/api/v1/traces/tasks/task-1/spans", "Bearer flm-test"),
            ("http://root:8000/api/v1/traces/tasks/task-1/lineage", "Bearer flm-test"),
        ]

    def test_no_server_url_uploads_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _traces(tmp_path)

        def post(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("nothing uploads without a server URL")

        monkeypatch.setattr(checkpoints.requests, "post", post)

        assert checkpoints.maybe_upload_traces(_task(), tmp_path, None) == []

    def test_a_failed_upload_is_left_out_and_the_rest_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _traces(tmp_path)

        class _Response:
            def __init__(self, url: str) -> None:
                self.url = url

            def raise_for_status(self) -> None:
                if self.url.endswith("/spans"):
                    raise checkpoints.requests.HTTPError("503")

        monkeypatch.setattr(
            checkpoints.requests, "post", lambda url, **kwargs: _Response(url)
        )

        assert checkpoints.maybe_upload_traces(
            _task(), tmp_path, "http://root:8000"
        ) == ["lineage"]


@pytest.mark.parametrize(
    ("url", "expects_auth"),
    [
        ("https://flowmesh.example/api/v1/results/tsk-1/files/model.bin", True),
        ("https://external.example/model.bin", False),
    ],
)
def test_download_and_unpack_only_authenticates_flowmesh_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expects_auth: bool,
) -> None:
    monkeypatch.setenv("FLOWMESH_API_KEY", "flm-test")
    monkeypatch.setenv("FLOWMESH_BASE_URL", "https://flowmesh.example")
    captured_calls: list[dict[str, Any]] = []

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b"checkpoint"]

    def _fake_get(
        request_url: str,
        *,
        headers: dict[str, str],
        stream: bool,
        timeout: float,
    ) -> _Response:
        captured_calls.append(
            {
                "url": request_url,
                "headers": headers,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return _Response()

    monkeypatch.setattr(checkpoints.requests, "get", _fake_get)

    path = checkpoints.download_and_unpack({"url": url, "timeoutSec": 12}, tmp_path)

    assert path.read_bytes() == b"checkpoint"
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["timeout"] == 12
    assert call["stream"] is True
    if expects_auth:
        assert call["headers"]["Authorization"] == "Bearer flm-test"
    else:
        assert "Authorization" not in call["headers"]

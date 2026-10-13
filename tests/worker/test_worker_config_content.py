from pathlib import Path

import pytest

from shared.content.wire import CHUNK_BYTES
from worker.config import WorkerConfig


@pytest.fixture(autouse=True)
def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "localhost:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.setenv("WORKER_ALIAS", "w")
    monkeypatch.setenv("WORKER_TOKEN", "managed-token")


def test_a_transfer_window_below_one_frame_is_held_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window under one frame would carry one frame at a time."""
    monkeypatch.setenv("CONTENT_TRANSFER_WINDOW_BYTES", "1024")

    assert WorkerConfig.from_env().content_transfer_window_bytes == CHUNK_BYTES


def test_a_worker_reads_the_peer_connect_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NETWORK_PLANE_CONNECT_BUDGET_SEC", "2.5")

    assert WorkerConfig.from_env().peer_connect_budget_sec == 2.5


@pytest.mark.parametrize("configured", ["0", "-1", "nonsense"])
def test_a_connect_budget_that_is_not_positive_is_the_default(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    monkeypatch.setenv("NETWORK_PLANE_CONNECT_BUDGET_SEC", configured)

    assert WorkerConfig.from_env().peer_connect_budget_sec == 5.0

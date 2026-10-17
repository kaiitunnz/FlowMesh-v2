"""A worker registers each multipart upload with control and waits for its verdict."""

import logging
import threading
from typing import Any, cast
from unittest import mock

import pytest

from shared.content import ObjectStoreConfig
from shared.content.multipart import (
    CONTENT_PROTOCOL,
    MULTIPART_ACK_FRAME_KIND,
    MultipartDoomed,
    MultipartUnowned,
    MultipartUnregistered,
    finalizing,
)
from shared.content.s3_store import S3ObjectStore
from shared.schemas.worker import WorkerCapabilities
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from worker.content.access import ContentAccessRegistry
from worker.content.multipart import MultipartRegistrar
from worker.content.plane import WorkerContentPlane
from worker.supervisor_client import SupervisorClient


class _Control:
    """Answers each registration it is sent with the next verdict, or not at all."""

    def __init__(self, *verdicts: str | None) -> None:
        self.verdicts = list(verdicts)
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.registrar: MultipartRegistrar | None = None

    def send(self, task_id: str, payload: dict[str, Any]) -> None:
        self.sent.append((task_id, dict(payload)))
        verdict = self.verdicts.pop(0) if self.verdicts else None
        if verdict is not None:
            assert self.registrar is not None
            frame = {"upload_id": payload["upload_id"], "verdict": verdict}
            threading.Thread(target=self.registrar.deliver, args=(frame,)).start()


def _registrar(control: _Control, wait_sec: float = 2.0) -> MultipartRegistrar:
    registrar = MultipartRegistrar(control.send, attempts=3, wait_sec=wait_sec)
    control.registrar = registrar
    return registrar


def test_an_acknowledged_registration_returns() -> None:
    control = _Control("ack")

    _registrar(control).register("tsk-1", "org", "org/sha256/a", "up-1")

    assert control.sent == [
        (
            "tsk-1",
            {
                "task_id": "tsk-1",
                "scope": "org",
                "key": "org/sha256/a",
                "upload_id": "up-1",
            },
        )
    ]


def test_a_lost_answer_is_asked_for_again() -> None:
    control = _Control(None, "ack")

    _registrar(control, wait_sec=0.05).register("tsk-1", "org", "k", "up-1")

    assert len(control.sent) == 2


def test_no_answer_leaves_the_upload_unregistered() -> None:
    control = _Control()

    with pytest.raises(MultipartUnregistered):
        _registrar(control, wait_sec=0.01).register("tsk-1", "org", "k", "up-1")

    assert len(control.sent) == 3


@pytest.mark.parametrize(
    "verdict, raised",
    [("doomed", MultipartDoomed), ("unowned", MultipartUnowned)],
)
def test_a_refusal_raises_what_the_store_acts_on(
    verdict: str, raised: type[Exception]
) -> None:
    control = _Control(verdict)

    with pytest.raises(raised):
        _registrar(control).register("tsk-1", "org", "k", "up-1")


def test_a_write_materializing_a_finalization_names_it() -> None:
    control = _Control("ack")

    with finalizing("idm-1"):
        _registrar(control).register("tsk-1", "org", "k", "up-1")

    assert control.sent[0][1]["idempotency_key"] == "idm-1"


def test_a_verdict_for_no_waiting_upload_is_dropped() -> None:
    registrar = MultipartRegistrar(lambda task_id, payload: None)

    registrar.deliver({"upload_id": "up-x", "verdict": "ack"})
    registrar.deliver({"upload_id": "up-x", "verdict": "bogus"})


def test_the_plane_routes_control_verdicts_to_the_registrar() -> None:
    control = _Control()
    registrar = _registrar(control, wait_sec=2.0)
    plane = WorkerContentPlane(
        None,
        ContentAccessRegistry(ObjectStoreConfig()),
        multipart=registrar,
    )
    done = threading.Event()

    def register() -> None:
        registrar.register("tsk-1", "org", "k", "up-1")
        done.set()

    thread = threading.Thread(target=register)
    thread.start()
    while not control.sent:
        threading.Event().wait(0.01)
    plane.route(MULTIPART_ACK_FRAME_KIND, {"upload_id": "up-1", "verdict": "ack"})
    thread.join(5)

    assert done.is_set()


def test_a_task_store_registers_its_uploads_under_its_own_task_and_scope() -> None:
    registered: list[tuple[str, str, str, str]] = []
    access = ContentAccessRegistry(
        ObjectStoreConfig(backend="s3", bucket="b"),
        register_multipart=lambda *args: registered.append(args),
    )

    store = access._open(("tsk-1", "org"))
    assert isinstance(store, S3ObjectStore)
    assert store._on_initiated is not None
    store._on_initiated("org/sha256/a", "up-1")

    assert registered == [("tsk-1", "org", "org/sha256/a", "up-1")]


def test_a_worker_reports_its_content_protocol_at_registration() -> None:
    client = SupervisorClient(
        worker_token="fm-worker-0.deadbeef",
        owner_principal=None,
        grpc_target="localhost:50051",
        worker_namespace="ns",
        worker_cluster="cl",
        worker_alias="al",
        logger=logging.getLogger("test"),
    )
    sent: list[dict[str, Any]] = []

    def register(meta: dict[str, Any]) -> None:
        sent.append(meta)
        client._worker_id = "wrk-1"

    with mock.patch.object(client, "_register_grpc", side_effect=register):
        client.register(
            WorkerStatus.IDLE,
            "2026-01-01T00:00:00Z",
            1,
            {},
            cast(WorkerHardware, mock.Mock()),
            WorkerCapabilities(),
            None,
            [],
            0.0,
        )

    assert sent[0]["content_protocol"] == str(CONTENT_PROTOCOL)

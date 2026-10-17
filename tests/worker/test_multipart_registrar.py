"""A worker registers each multipart upload with control and waits for its verdict."""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from shared.content import ObjectStoreConfig, WriteCancelled, content_digest
from shared.content.multipart import (
    CONTENT_PROTOCOL,
    MULTIPART_ACK_FRAME_KIND,
    MultipartDoomed,
    MultipartUnowned,
    finalizing,
)
from shared.content.s3_store import S3ObjectStore
from shared.schemas.worker import WorkerCapabilities
from shared.tasks.worker_message import WorkerHardware, WorkerStatus
from tests.shared.fake_s3 import FakeS3
from worker.content.access import ContentAccessRegistry
from worker.content.multipart import MultipartRegistrar
from worker.content.plane import WorkerContentPlane
from worker.supervisor_client import SupervisorClient

_PART = 5 << 20
_DATA = b"x" * (2 * _PART + 7)


class _Time:
    """A clock that a wait for an unset event advances by the whole timeout."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def wait(self, event: threading.Event, timeout: float) -> bool:
        assert self.now < 10_000, "a registration outwaited every bound in the test"
        if not event.is_set():
            self.now += timeout
        return event.is_set()


class _Control:
    """Answers each registration it is sent with the next verdict, or not at all, and
    answers every one with ``ack`` once the clock reaches ``up_at``."""

    def __init__(
        self, *verdicts: str | None, time: _Time | None = None, up_at: float = -1.0
    ) -> None:
        self.verdicts = list(verdicts)
        self.time = time
        self.up_at = up_at
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.sent_at: list[float] = []
        self.registrar: MultipartRegistrar | None = None
        self.cv = threading.Condition()

    def send(self, task_id: str, payload: dict[str, Any]) -> None:
        with self.cv:
            self.sent.append((task_id, dict(payload)))
            self.cv.notify_all()
        if self.time is not None:
            self.sent_at.append(self.time.now)
        verdict = self.verdicts.pop(0) if self.verdicts else None
        if self.time is not None and 0 <= self.up_at <= self.time.now:
            verdict = "ack"
        if verdict is not None:
            assert self.registrar is not None
            self.registrar.deliver(
                {"upload_id": payload["upload_id"], "verdict": verdict}
            )

    def wait_sent(self, count: int) -> None:
        with self.cv:
            assert self.cv.wait_for(lambda: len(self.sent) >= count, 5.0), "no send"


def _registrar(control: _Control, time: _Time | None = None) -> MultipartRegistrar:
    time = time or control.time or _Time()
    registrar = MultipartRegistrar(control.send, clock=time.clock, wait=time.wait)
    control.registrar = registrar
    return registrar


def _store(registrar: MultipartRegistrar, s3: FakeS3) -> S3ObjectStore:
    def initiated(
        key: str, upload_id: str, cancelled: Callable[[], bool] | None
    ) -> None:
        registrar.register("tsk-1", "org", key, upload_id, cancelled)

    return S3ObjectStore(
        s3, "bucket", part_bytes=_PART, upload_concurrency=2, on_initiated=initiated
    )


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

    _registrar(control).register("tsk-1", "org", "k", "up-1")

    assert len(control.sent) == 2


@pytest.mark.parametrize("from_file", [False, True], ids=["bytes", "file"])
def test_a_large_write_waits_out_a_control_outage_and_completes(
    tmp_path: Path, from_file: bool
) -> None:
    time = _Time()
    control = _Control(time=time, up_at=40.0)
    s3 = FakeS3()
    store = _store(_registrar(control), s3)

    if from_file:
        path = tmp_path / "src.bin"
        path.write_bytes(_DATA)
        reference = store.write_file("org", path, cancelled=lambda: False)
    else:
        reference = store.write("org", _DATA)

    # Asked again at a pace that doubles to its cap, through an outage longer than
    # three five-second asks.
    assert control.sent_at == [0.0, 5.0, 15.0, 35.0, 65.0]
    assert s3.objects[f"org/sha256/{content_digest(_DATA)}"] == _DATA
    assert reference.size_bytes == len(_DATA)
    assert s3.aborted == []


def test_a_write_cancelled_while_control_is_out_ends_as_cancelled(
    tmp_path: Path,
) -> None:
    time = _Time()
    control = _Control(time=time)
    s3 = FakeS3()
    path = tmp_path / "src.bin"
    path.write_bytes(_DATA)

    with pytest.raises(WriteCancelled):
        _store(_registrar(control), s3).write_file(
            "org", path, cancelled=lambda: time.now >= 100.0
        )

    assert 100.0 <= time.now < 101.0
    assert s3.aborted == ["up-1"]
    assert s3.uploads == {} and s3.objects == {}


def test_an_abandoned_registration_ends_its_write_as_cancelled() -> None:
    control = _Control()
    registrar = MultipartRegistrar(control.send)
    control.registrar = registrar
    raised: list[BaseException] = []

    def register() -> None:
        try:
            registrar.register("tsk-1", "org", "k", "up-1")
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=register, daemon=True)
    thread.start()
    control.wait_sent(1)
    registrar.abandon()
    thread.join(5)

    assert not thread.is_alive()
    assert len(raised) == 1 and isinstance(raised[0], WriteCancelled)


def test_a_stopping_plane_abandons_the_registrations_it_waits_on() -> None:
    control = _Control()
    registrar = MultipartRegistrar(control.send)
    control.registrar = registrar
    plane = WorkerContentPlane(
        None, ContentAccessRegistry(ObjectStoreConfig()), multipart=registrar
    )
    raised: list[BaseException] = []

    def register() -> None:
        try:
            registrar.register("tsk-1", "org", "k", "up-1")
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=register, daemon=True)
    thread.start()
    control.wait_sent(1)
    plane.stop()
    thread.join(5)

    assert len(raised) == 1 and isinstance(raised[0], WriteCancelled)


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
    registrar = MultipartRegistrar(control.send)
    control.registrar = registrar
    plane = WorkerContentPlane(
        None,
        ContentAccessRegistry(ObjectStoreConfig()),
        multipart=registrar,
    )
    done = threading.Event()

    def register() -> None:
        registrar.register("tsk-1", "org", "k", "up-1")
        done.set()

    thread = threading.Thread(target=register, daemon=True)
    thread.start()
    control.wait_sent(1)
    plane.route(MULTIPART_ACK_FRAME_KIND, {"upload_id": "up-1", "verdict": "ack"})
    thread.join(5)

    assert done.is_set()


def test_a_task_store_registers_its_uploads_under_its_own_task_and_scope() -> None:
    registered: list[tuple[Any, ...]] = []
    access = ContentAccessRegistry(
        ObjectStoreConfig(backend="s3", bucket="b"),
        register_multipart=lambda *args: registered.append(args),
    )

    store = access._open(("tsk-1", "org"))
    assert isinstance(store, S3ObjectStore)
    assert store._on_initiated is not None

    def cancelled() -> bool:
        return False

    store._on_initiated("org/sha256/a", "up-1", cancelled)

    assert registered == [("tsk-1", "org", "org/sha256/a", "up-1", cancelled)]


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

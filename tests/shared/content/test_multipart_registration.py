"""A multipart write registers its exact UploadId before it sends any part."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shared.content import ContentStoreError, WriteCancelled, content_digest
from shared.content.multipart import MultipartDoomed, MultipartUnowned
from shared.content.s3_store import S3ObjectStore
from tests.shared.fake_s3 import FakeS3

_PART = 5 << 20
_DATA = b"x" * (2 * _PART + 7)


class _OrderedS3(FakeS3):
    """A fake S3 that logs each multipart call in order."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def create_multipart_upload(self, **kw: Any) -> Any:
        reply = super().create_multipart_upload(**kw)
        self.calls.append(("create", reply["UploadId"]))
        return reply

    def upload_part(self, **kw: Any) -> Any:
        self.calls.append(("part", kw["UploadId"]))
        return super().upload_part(**kw)

    def abort_multipart_upload(self, **kw: Any) -> Any:
        self.calls.append(("abort", kw["UploadId"]))
        return super().abort_multipart_upload(**kw)


class _Registrar:
    def __init__(self, s3: _OrderedS3, *answers: BaseException | None) -> None:
        self._s3 = s3
        self._answers = list(answers)
        self.registered: list[tuple[str, str]] = []
        self.cancelled: list[Callable[[], bool] | None] = []

    def __call__(
        self, key: str, upload_id: str, cancelled: Callable[[], bool] | None
    ) -> None:
        self._s3.calls.append(("register", upload_id))
        self.registered.append((key, upload_id))
        self.cancelled.append(cancelled)
        answer = self._answers.pop(0) if self._answers else None
        if answer is not None:
            raise answer


def _store(s3: _OrderedS3, registrar: _Registrar) -> S3ObjectStore:
    return S3ObjectStore(
        s3, "bucket", part_bytes=_PART, upload_concurrency=2, on_initiated=registrar
    )


@pytest.mark.parametrize("from_file", [False, True], ids=["bytes", "file"])
def test_the_upload_is_registered_before_any_part(
    tmp_path: Path, from_file: bool
) -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3)
    store = _store(s3, registrar)

    if from_file:
        path = tmp_path / "src.bin"
        path.write_bytes(_DATA)
        reference = store.write_file("org", path)
    else:
        reference = store.write("org", _DATA)

    assert s3.calls[:2] == [("create", "up-1"), ("register", "up-1")]
    assert [call for call, _ in s3.calls[2:]] == ["part"] * 3
    key = f"org/sha256/{content_digest(_DATA)}"
    assert registrar.registered == [(key, "up-1")]
    assert s3.objects[key] == _DATA
    assert reference.size_bytes == len(_DATA)


def test_a_single_put_registers_nothing() -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3)

    _store(s3, registrar).write("org", b"small")

    assert registrar.registered == []
    assert s3.created == 0


def test_a_doomed_upload_is_aborted_by_its_id_and_sent_under_a_new_one() -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3, MultipartDoomed("aborting"))

    _store(s3, registrar).write("org", _DATA)

    assert s3.calls[:5] == [
        ("create", "up-1"),
        ("register", "up-1"),
        ("abort", "up-1"),
        ("create", "up-2"),
        ("register", "up-2"),
    ]
    assert {upload for call, upload in s3.calls if call == "part"} == {"up-2"}
    assert s3.objects[f"org/sha256/{content_digest(_DATA)}"] == _DATA


def test_an_upload_doomed_every_time_fails_after_bounded_attempts() -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3, *(MultipartDoomed("aborting") for _ in range(5)))

    with pytest.raises(ContentStoreError, match="doomed"):
        _store(s3, registrar).write("org", _DATA)

    assert s3.created == 3
    assert s3.aborted == ["up-1", "up-2", "up-3"]
    assert not any(call == "part" for call, _ in s3.calls)


@pytest.mark.parametrize(
    "failure, raised",
    [
        (WriteCancelled("cancelled"), WriteCancelled),
        (MultipartUnowned("no owner"), MultipartUnowned),
        (ConnectionError("stream down"), ContentStoreError),
    ],
    ids=["cancelled", "unowned", "unexpected"],
)
def test_a_registration_that_fails_aborts_the_upload_and_sends_nothing(
    failure: Exception, raised: type[Exception]
) -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3, failure)

    with pytest.raises(raised):
        _store(s3, registrar).write("org", _DATA)

    assert s3.calls == [("create", "up-1"), ("register", "up-1"), ("abort", "up-1")]
    assert s3.uploads == {}
    assert s3.objects == {}


def test_an_interrupted_registration_aborts_the_upload() -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _store(s3, registrar).write("org", _DATA)

    assert s3.aborted == ["up-1"]


def test_an_object_already_stored_registers_nothing() -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3)
    store = _store(s3, registrar)
    store.write("org", _DATA)
    registrar.registered.clear()

    store.write("org", _DATA)

    assert registrar.registered == []


def test_a_file_write_registers_under_its_own_cancellation(tmp_path: Path) -> None:
    s3 = _OrderedS3()
    registrar = _Registrar(s3)
    path = tmp_path / "src.bin"
    path.write_bytes(_DATA)

    def cancelled() -> bool:
        return False

    _store(s3, registrar).write_file("org", path, cancelled=cancelled)

    assert registrar.cancelled == [cancelled]

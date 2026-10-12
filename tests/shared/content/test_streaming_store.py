"""Streaming writes and reads through the shared store, conditional on the key."""

import base64
import hashlib
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from shared.content import (
    ContentHydrationError,
    ContentStoreError,
    ContentUnavailable,
    FabricObjectStore,
    SharedFilesystemObjectStore,
    WriteCancelled,
    content_digest,
    reference_for,
)
from shared.content.s3_store import S3ObjectStore

_PART = 5 << 20


def _error(code: str, status: int, op: str = "PutObject") -> ClientError:
    response: Any = {
        "Error": {"Code": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, op)


def _b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode()


class _Body:
    def __init__(self, data: bytes, *, fail_after: int | None = None) -> None:
        self._data = data
        self._fail_after = fail_after

    def read(self) -> bytes:
        return self._data

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self._data), chunk_size):
            if self._fail_after is not None and start >= self._fail_after:
                raise ConnectionResetError("reset mid-body")
            yield self._data[start : start + chunk_size]

    def close(self) -> None:
        pass


class FakeS3:
    """An in-memory S3 that enforces conditional writes and per-part checksums."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.aborted: list[str] = []
        self.puts = 0
        self.fail_part: int | None = None
        self.body_fail_after: int | None = None
        self._lock = threading.Lock()
        self._next = 0

    def put_object(self, **kw: Any) -> Any:
        data = kw["Body"]
        if kw.get("ChecksumSHA256") not in (None, _b64(data)):
            raise _error("XAmzContentChecksumMismatch", 400)
        with self._lock:
            self.puts += 1
            if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.objects:
                raise _error("PreconditionFailed", 412)
            self.objects[kw["Key"]] = bytes(data)
        return {}

    def get_object(self, **kw: Any) -> Any:
        if kw["Key"] not in self.objects:
            raise _error("NoSuchKey", 404, "GetObject")
        return {"Body": _Body(self.objects[kw["Key"]], fail_after=self.body_fail_after)}

    def head_object(self, **kw: Any) -> Any:
        if kw["Key"] not in self.objects:
            raise _error("404", 404, "HeadObject")
        return {"ContentLength": len(self.objects[kw["Key"]])}

    def create_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            self._next += 1
            upload_id = f"up-{self._next}"
            self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, **kw: Any) -> Any:
        data = kw["Body"]
        if self.fail_part is not None and kw["PartNumber"] == self.fail_part:
            raise _error("InternalError", 500, "UploadPart")
        if kw["ChecksumSHA256"] != _b64(data):
            raise _error("XAmzContentChecksumMismatch", 400, "UploadPart")
        with self._lock:
            self.uploads[kw["UploadId"]][kw["PartNumber"]] = bytes(data)
        return {"ETag": f"etag-{kw['PartNumber']}"}

    def complete_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            if kw["UploadId"] not in self.uploads:
                raise _error("NoSuchUpload", 404, "CompleteMultipartUpload")
            parts = self.uploads[kw["UploadId"]]
            if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.objects:
                raise _error("PreconditionFailed", 412, "CompleteMultipartUpload")
            numbers = [part["PartNumber"] for part in kw["MultipartUpload"]["Parts"]]
            assert numbers == sorted(numbers)
            self.objects[kw["Key"]] = b"".join(parts[n] for n in numbers)
            del self.uploads[kw["UploadId"]]
        return {}

    def abort_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            self.uploads.pop(kw["UploadId"], None)
            self.aborted.append(kw["UploadId"])
        return {}


def _write_file(tmp_path: Path, data: bytes, name: str = "src.bin") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.fixture(params=["s3", "filesystem"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> FabricObjectStore:
    if request.param == "s3":
        return S3ObjectStore(FakeS3(), "bucket", part_bytes=_PART, upload_concurrency=3)
    return SharedFilesystemObjectStore(tmp_path / "shared")


@pytest.mark.parametrize("size", [0, 1, _PART, _PART + 1, 3 * _PART + 7])
def test_a_written_file_reads_back_verified(
    store: FabricObjectStore, tmp_path: Path, size: int
) -> None:
    data = os.urandom(size)
    reference = store.write_file("org", _write_file(tmp_path, data))

    assert reference == reference_for("org", data)
    assert b"".join(store.open(reference, chunk_bytes=4096)) == data
    store.hydrate_to(reference, tmp_path / "out" / "copy.bin")
    assert (tmp_path / "out" / "copy.bin").read_bytes() == data
    assert store.hydrate(reference) == data


@pytest.mark.parametrize("size", [10, 3 * _PART + 7])
def test_bytes_and_file_writes_name_the_same_object(
    store: FabricObjectStore, tmp_path: Path, size: int
) -> None:
    data = os.urandom(size)
    assert store.write("org", data) == store.write_file(
        "org", _write_file(tmp_path, data)
    )


@pytest.mark.parametrize("size", [10, 3 * _PART + 7])
def test_a_second_write_of_one_object_leaves_the_first(size: int) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    data = os.urandom(size)
    first = store.write("org", data)

    assert store.write("org", data) == first
    assert len(client.objects) == 1
    assert client.uploads == {}


def test_a_large_object_already_stored_is_not_sent_again() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    data = os.urandom(2 * _PART)
    store.write("org", data)
    created = client._next

    store.write("org", data)

    assert client._next == created


def test_a_commit_refused_by_a_racing_writer_is_aborted() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    data = os.urandom(2 * _PART)
    store.write("org", data)
    # The object lands between the presence check and the commit.
    head = client.head_object
    calls: list[int] = []

    def head_after_refusal(**kw: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise _error("404", 404, "HeadObject")
        return head(**kw)

    client.head_object = head_after_refusal  # type: ignore[method-assign]

    store.write("org", data)

    assert client.uploads == {}
    assert len(client.aborted) == 1


def test_a_commit_racing_another_writer_is_retried() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    original = client.put_object
    conflicts: list[int] = []

    def conflicting(**kw: Any) -> Any:
        if not conflicts:
            conflicts.append(1)
            raise _error("ConditionalRequestConflict", 409)
        return original(**kw)

    client.put_object = conflicting  # type: ignore[method-assign]

    reference = store.write("org", b"payload")

    assert store.hydrate(reference) == b"payload"


def test_a_part_whose_session_expired_is_sent_again_on_renewed_access(
    tmp_path: Path,
) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART, upload_concurrency=1)
    original = client.upload_part
    expired: list[int] = []

    def expiring(**kw: Any) -> Any:
        if kw["PartNumber"] == 2 and not expired:
            expired.append(1)
            raise _error("ExpiredToken", 400, "UploadPart")
        return original(**kw)

    client.upload_part = expiring  # type: ignore[method-assign]
    data = os.urandom(3 * _PART)

    reference = store.write_file("org", _write_file(tmp_path, data))

    assert expired == [1]
    assert store.hydrate(reference) == data


def test_an_existing_object_of_the_wrong_size_is_refused() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    data = b"twelve bytes"
    reference = store.write("org", data)
    client.objects[next(iter(client.objects))] = b"truncated"

    with pytest.raises(ContentStoreError, match="already stored"):
        store.write("org", data)
    with pytest.raises(ContentHydrationError):
        store.hydrate(reference)


def test_a_failed_part_aborts_the_upload(tmp_path: Path) -> None:
    client = FakeS3()
    client.fail_part = 2
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)

    with pytest.raises(ContentStoreError):
        store.write_file("org", _write_file(tmp_path, os.urandom(3 * _PART)))

    assert client.objects == {}
    assert client.uploads == {}
    assert client.aborted


def test_a_cancelled_upload_is_aborted(tmp_path: Path) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART, upload_concurrency=1)
    sent: list[int] = []
    original = client.upload_part

    def counting(**kw: Any) -> Any:
        sent.append(kw["PartNumber"])
        return original(**kw)

    client.upload_part = counting  # type: ignore[method-assign]

    with pytest.raises(WriteCancelled):
        store.write_file(
            "org",
            _write_file(tmp_path, os.urandom(4 * _PART)),
            cancelled=lambda: len(sent) >= 2,
        )

    assert client.objects == {}
    assert client.uploads == {}
    assert len(sent) == 2


def test_a_file_changed_between_hash_and_send_is_refused(tmp_path: Path) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART, upload_concurrency=1)
    path = _write_file(tmp_path, os.urandom(3 * _PART))
    original = client.create_multipart_upload

    def mutate_then_create(**kw: Any) -> Any:
        with path.open("r+b") as fh:
            fh.seek(_PART + 3)
            fh.write(b"changed")
        return original(**kw)

    client.create_multipart_upload = mutate_then_create  # type: ignore[method-assign]

    with pytest.raises(ContentStoreError):
        store.write_file("org", path)
    assert client.objects == {}
    assert client.uploads == {}


def test_a_read_withholds_its_last_chunk_from_bytes_that_do_not_verify(
    store: FabricObjectStore,
) -> None:
    data = os.urandom(3 * 4096)
    good = store.write("org", data)
    forged = good.model_copy(update={"content_digest": content_digest(b"other")})

    received: list[bytes] = []
    with pytest.raises(ContentHydrationError):
        for chunk in store.open(forged, chunk_bytes=4096):
            received.append(chunk)
    assert sum(map(len, received)) < len(data)


def test_a_hydration_that_fails_leaves_nothing_behind(
    store: FabricObjectStore, tmp_path: Path
) -> None:
    good = store.write("org", os.urandom(9000))
    short = good.model_copy(update={"size_bytes": 8000})
    dest = tmp_path / "out" / "copy.bin"

    with pytest.raises(ContentHydrationError):
        store.hydrate_to(short, dest)
    assert not dest.exists()
    assert list((tmp_path / "out").iterdir()) == []


def test_a_body_broken_mid_read_is_transient() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    reference = store.write("org", os.urandom(20000))
    client.body_fail_after = 8192

    with pytest.raises(ContentUnavailable):
        list(store.open(reference, chunk_bytes=4096))


def test_concurrent_filesystem_writers_commit_one_object(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    data = os.urandom(1 << 20)
    errors: list[BaseException] = []

    def write() -> None:
        try:
            store.write("org", data)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    objects = [p for p in (tmp_path / "shared").rglob("*") if p.is_file()]
    assert len(objects) == 1
    assert objects[0].read_bytes() == data


def test_a_cancelled_filesystem_write_leaves_no_staging(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    calls: list[int] = []

    def cancelled() -> bool:
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(WriteCancelled):
        store.write_file(
            "org",
            _write_file(tmp_path, os.urandom(3 << 20)),
            cancelled=cancelled,
        )
    assert [p for p in (tmp_path / "shared").rglob("*") if p.is_file()] == []


def test_a_truncated_filesystem_object_is_refused_on_rewrite(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    data = os.urandom(5000)
    reference = store.write("org", data)
    (path,) = [p for p in (tmp_path / "shared").rglob("*") if p.is_file()]
    path.write_bytes(data[:100])

    with pytest.raises(ContentStoreError, match="already stored"):
        store.write("org", data)
    with pytest.raises(ContentHydrationError):
        b"".join(store.open(reference))


def test_a_missing_object_is_a_permanent_miss(store: FabricObjectStore) -> None:
    with pytest.raises(ContentHydrationError) as raised:
        list(store.open(reference_for("org", b"never written")))
    assert not isinstance(raised.value, ContentUnavailable)


def test_a_commit_whose_answer_was_lost_settles_on_the_object(tmp_path: Path) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "bucket", part_bytes=_PART)
    original = client.complete_multipart_upload
    answered: list[int] = []

    def lost_answer(**kw: Any) -> Any:
        if not answered:
            answered.append(1)
            original(**kw)
            raise _error("ConditionalRequestConflict", 409, "CompleteMultipartUpload")
        return original(**kw)

    client.complete_multipart_upload = lost_answer  # type: ignore[method-assign]
    data = os.urandom(2 * _PART)

    reference = store.write_file("org", _write_file(tmp_path, data))

    assert store.hydrate(reference) == data


def test_a_store_without_conditional_writes_is_named_as_such() -> None:
    client = FakeS3()

    def unsupported(**kw: Any) -> Any:
        raise _error("NotImplemented", 501)

    client.put_object = unsupported  # type: ignore[method-assign]

    with pytest.raises(ContentStoreError, match="conditional writes"):
        S3ObjectStore(client, "bucket").write("org", b"payload")

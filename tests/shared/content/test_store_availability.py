"""A store read that cannot reach the store is told apart from one naming nothing."""

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from shared.content import (
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    SharedFilesystemObjectStore,
)
from shared.content.s3_store import S3ObjectStore
from tests.shared.fake_s3 import UnusedWrites

_REF = ContentReference(
    authorization_scope="org", content_digest="a" * 64, size_bytes=1
)


class _Client(UnusedWrites):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def put_object(self, **kwargs: Any) -> Any:
        return {}

    def get_object(self, **kwargs: Any) -> Any:
        raise self._error


def _client_error(code: str, status: int) -> ClientError:
    response: Any = {
        "Error": {"Code": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "GetObject")


@pytest.mark.parametrize(
    "error", [_client_error("NoSuchKey", 404), _client_error("NotFound", 404)]
)
def test_an_absent_object_is_a_permanent_miss(error: Exception) -> None:
    with pytest.raises(ContentHydrationError) as raised:
        S3ObjectStore(_Client(error), "bucket").fetch(_REF)
    assert not isinstance(raised.value, ContentUnavailable)


@pytest.mark.parametrize(
    "error",
    [
        _client_error("SlowDown", 503),
        _client_error("InternalError", 500),
        _client_error("AccessDenied", 403),
        EndpointConnectionError(endpoint_url="http://store"),
        TimeoutError("read timed out"),
    ],
)
def test_an_unreachable_store_is_transient(error: Exception) -> None:
    with pytest.raises(ContentUnavailable):
        S3ObjectStore(_Client(error), "bucket").fetch(_REF)


def test_a_body_that_breaks_mid_read_is_transient() -> None:
    class _Body(BytesIO):
        def read(self, *args: Any) -> bytes:
            raise ConnectionResetError("reset")

    class _Reading(_Client):
        def get_object(self, **kwargs: Any) -> Any:
            return {"Body": _Body()}

    with pytest.raises(ContentUnavailable):
        S3ObjectStore(_Reading(TimeoutError()), "bucket").fetch(_REF)


def test_a_missing_file_is_permanent_and_an_unreadable_one_transient(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path)
    with pytest.raises(ContentHydrationError) as raised:
        store.fetch(_REF)
    assert not isinstance(raised.value, ContentUnavailable)

    reference = store.write("org", b"x")
    (target,) = tmp_path.rglob(reference.content_digest)
    target.chmod(0)
    try:
        with pytest.raises(ContentUnavailable):
            store.fetch(reference)
    finally:
        target.chmod(0o644)


class _Heading(_Client):
    def __init__(self, error: Exception | None, length: int = 7) -> None:
        super().__init__(error or TimeoutError())
        self._head_error = error
        self._length = length

    def get_object(self, **kwargs: Any) -> Any:
        raise AssertionError("a size check read the object")

    def head_object(self, **kwargs: Any) -> Any:
        if self._head_error is not None:
            raise self._head_error
        return {"ContentLength": self._length}


def test_a_stored_size_reads_no_bytes() -> None:
    assert S3ObjectStore(_Heading(None, 7), "bucket").stored_size(_REF) == 7


def test_a_stored_size_tells_absent_from_unreachable(tmp_path: Path) -> None:
    with pytest.raises(ContentHydrationError) as absent:
        S3ObjectStore(_Heading(_client_error("404", 404)), "bucket").stored_size(_REF)
    assert not isinstance(absent.value, ContentUnavailable)
    with pytest.raises(ContentUnavailable):
        S3ObjectStore(
            _Heading(EndpointConnectionError(endpoint_url="http://store")), "bucket"
        ).stored_size(_REF)

    store = SharedFilesystemObjectStore(tmp_path)
    assert store.stored_size(store.write("org", b"abc")) == 3
    with pytest.raises(ContentHydrationError) as missing:
        store.stored_size(_REF)
    assert not isinstance(missing.value, ContentUnavailable)

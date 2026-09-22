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

_REF = ContentReference(
    authorization_scope="org", content_digest="a" * 64, size_bytes=1
)


class _Client:
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

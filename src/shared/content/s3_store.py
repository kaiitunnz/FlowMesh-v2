"""The shared durable store the fabric's content actually lives in.

An S3-compatible object store — MinIO, cloud S3, anything speaking the same API — holds
every content object as the source of truth. A worker writes an object here before it
reports the reference naming it, so a reference that reaches any binding names bytes
that already survive the worker that produced them; a worker that needs an object and
does not have it reads from here. Workers hold copies to save that read, but a copy is
only ever a copy: this is where the object is.

The key is the reference's own identity — scope first, so a scope is a prefix an access
policy can be written against, then the algorithm and digest naming the bytes. A writer
hashes the bytes before it names them, so the key is known before anything is sent, and
every write is conditional on the key being free: a second write of the same object is
refused by the store and settles on the stored object once its size matches.

A large object goes up in parts, a bounded number at a time, each carrying the SHA-256
of its own bytes so the store refuses a part that arrives other than it was hashed. The
multipart checksum the store derives from those is a composite over parts and is never
taken for the object's digest.
"""

import base64
import concurrent.futures
import contextlib
import hashlib
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from botocore.exceptions import ClientError

from .config import DEFAULT_PART_BYTES, DEFAULT_UPLOAD_CONCURRENCY
from .reference import OCTET_STREAM, ContentReference
from .store import (
    STREAM_CHUNK_BYTES,
    ContentHydrationError,
    ContentStoreError,
    ContentUnavailable,
    FabricObjectStore,
    WriteCancelled,
    check_existing_size,
    check_range,
    new_hasher,
    reference_for,
    short_range,
    verified_chunks,
)

# The error codes an S3-compatible store answers with when the object is not there.
_MISSING_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
# The answer to a conditional write whose key is already taken.
_TAKEN_CODES = frozenset({"PreconditionFailed", "412"})
# The answer to a conditional write racing another write of the same key; the store
# asks the writer to try again, and the retry is refused or lands.
_CONFLICT_CODES = frozenset({"ConditionalRequestConflict", "409"})
# A session the store no longer honours, or never did, for the call it refused.
_CREDENTIAL_CODES = frozenset(
    {"ExpiredToken", "AccessDenied", "InvalidAccessKeyId", "TokenRefreshRequired"}
)
_CONFLICT_RETRIES = 5
_CONFLICT_BACKOFF_SEC = 0.2
# S3 numbers an upload's parts from 1 to this.
_MAX_PARTS = 10_000


class S3Client(Protocol):
    """The calls a content object needs from an S3-compatible client."""

    def put_object(self, **kwargs: Any) -> Any: ...

    def get_object(self, **kwargs: Any) -> Any: ...

    def head_object(self, **kwargs: Any) -> Any: ...

    def create_multipart_upload(self, **kwargs: Any) -> Any: ...

    def upload_part(self, **kwargs: Any) -> Any: ...

    def complete_multipart_upload(self, **kwargs: Any) -> Any: ...

    def abort_multipart_upload(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class _FileDigest:
    digest: str
    size: int
    part_bytes: int
    # The base64 SHA-256 of each part, in order.
    part_checksums: tuple[str, ...]


def _checksum(data: bytes | memoryview) -> str:
    return _b64(hashlib.sha256(data).digest())


def _b64(digest: bytes) -> str:
    return base64.b64encode(digest).decode()


class S3ObjectStore(FabricObjectStore):
    """Content objects in one bucket of an S3-compatible store."""

    def __init__(
        self,
        client: S3Client,
        bucket: str,
        *,
        prefix: str = "",
        part_bytes: int = DEFAULT_PART_BYTES,
        upload_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._part_bytes = part_bytes
        self._upload_concurrency = upload_concurrency

    def _key(self, reference: ContentReference) -> str:
        scope, algorithm, digest = reference.identity
        key = f"{scope}/{algorithm}/{digest}"
        return f"{self._prefix}/{key}" if self._prefix else key

    def write(
        self, scope: str, data: bytes, *, media_type: str = OCTET_STREAM
    ) -> ContentReference:
        reference = reference_for(scope, data, media_type=media_type)
        if len(data) <= self._part_bytes:
            self._put(reference, data)
            return reference
        part_bytes = self._part_size(len(data))
        view = memoryview(data)
        starts = range(0, len(data), part_bytes)
        self._upload(
            reference,
            [_checksum(view[start : start + part_bytes]) for start in starts],
            lambda index: data[index * part_bytes : (index + 1) * part_bytes],
            None,
        )
        return reference

    def write_file(
        self,
        scope: str,
        path: Path,
        *,
        media_type: str = OCTET_STREAM,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        digest = self._digest_file(path, cancelled)
        reference = ContentReference(
            authorization_scope=scope,
            content_digest=digest.digest,
            size_bytes=digest.size,
            media_type=media_type,
        )
        changed = ContentStoreError(f"{path.name} changed while it was written")
        if digest.size <= self._part_bytes:
            data = path.read_bytes()
            if (checksum := _checksum(data)) != digest.part_checksums[0]:
                raise changed
            self._put(reference, data, checksum)
            return reference
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:

            def read_part(index: int) -> bytes:
                # The bytes sent are the bytes hashed: a store that did not check a
                # part's checksum would otherwise keep a changed part under the digest.
                part = _pread_exactly(
                    fd,
                    min(digest.part_bytes, digest.size - index * digest.part_bytes),
                    index * digest.part_bytes,
                )
                if _checksum(part) != digest.part_checksums[index]:
                    raise changed
                return part

            self._upload(reference, digest.part_checksums, read_part, cancelled)
        finally:
            os.close(fd)
        return reference

    def _part_size(self, size: int) -> int:
        return max(self._part_bytes, math.ceil(size / _MAX_PARTS))

    def _digest_file(
        self, path: Path, cancelled: Callable[[], bool] | None
    ) -> _FileDigest:
        """Hash a file whole and per part in one read, before anything is sent."""
        size = path.stat().st_size
        part_bytes = self._part_size(size)
        whole = new_hasher()
        checksums: list[str] = []
        read = 0
        with path.open("rb") as fh:
            while True:
                if cancelled is not None and cancelled():
                    raise WriteCancelled(f"write of {path.name} was cancelled")
                part = hashlib.sha256()
                taken = 0
                while taken < part_bytes and (
                    chunk := fh.read(min(STREAM_CHUNK_BYTES, part_bytes - taken))
                ):
                    whole.update(chunk)
                    part.update(chunk)
                    taken += len(chunk)
                if taken == 0 and checksums:
                    break
                checksums.append(_b64(part.digest()))
                read += taken
                if taken < part_bytes:
                    break
        if read != size:
            raise ContentStoreError(f"{path.name} changed while it was hashed")
        return _FileDigest(
            digest=whole.hexdigest(),
            size=size,
            part_bytes=part_bytes,
            part_checksums=tuple(checksums),
        )

    def _put(
        self, reference: ContentReference, data: bytes, checksum: str | None = None
    ) -> None:
        try:
            _committing(
                lambda: self._client.put_object(
                    Bucket=self._bucket,
                    Key=self._key(reference),
                    Body=data,
                    ContentType=reference.media_type,
                    ChecksumSHA256=checksum or _checksum(data),
                    IfNoneMatch="*",
                )
            )
        except Exception as exc:
            if _is_taken(exc):
                self._check_existing(reference)
                return
            raise _write_error(exc) from exc

    def _upload(
        self,
        reference: ContentReference,
        checksums: Sequence[str],
        read_part: Callable[[int], bytes],
        cancelled: Callable[[], bool] | None,
    ) -> None:
        """Send an object in parts and commit it only where its key is free.

        Anything that stops the upload short of its commit — a failed part, a refused
        commit, a cancellation, the caller's own interruption — aborts it, so the store
        keeps no parts nobody will complete.
        """
        key = self._key(reference)
        if self._present(reference):
            return
        try:
            upload_id = self._client.create_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                ContentType=reference.media_type,
                ChecksumAlgorithm="SHA256",
            )["UploadId"]
        except Exception as exc:
            raise _write_error(exc) from exc
        try:
            parts = self._send_parts(key, upload_id, checksums, read_part, cancelled)
            _committing(
                lambda: self._client.complete_multipart_upload(
                    Bucket=self._bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                    IfNoneMatch="*",
                )
            )
        except BaseException as exc:
            self._abort(key, upload_id)
            if isinstance(exc, Exception) and (_is_taken(exc) or _is_gone(exc)):
                # A refused commit means the object is there; an upload gone by the time
                # its commit is retried means an earlier attempt of that commit landed
                # and its answer was lost. Either way the object decides.
                self._check_existing(reference)
                return
            if isinstance(exc, ContentStoreError) or not isinstance(exc, Exception):
                raise
            raise _write_error(exc) from exc

    def _send_parts(
        self,
        key: str,
        upload_id: str,
        checksums: Sequence[str],
        read_part: Callable[[int], bytes],
        cancelled: Callable[[], bool] | None,
    ) -> list[dict[str, Any]]:
        def send(index: int) -> dict[str, Any]:
            if cancelled is not None and cancelled():
                raise WriteCancelled(f"write of {key} was cancelled")
            body = read_part(index)
            response = self._client.upload_part(
                Bucket=self._bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=index + 1,
                Body=body,
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=checksums[index],
            )
            return {
                "PartNumber": index + 1,
                "ETag": response["ETag"],
                "ChecksumSHA256": checksums[index],
            }

        # Parts are submitted only as earlier ones finish, so no more than the
        # concurrency's worth of part buffers is ever held.
        parts: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(self._upload_concurrency) as pool:
            pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
            index = 0
            try:
                while index < len(checksums) or pending:
                    while index < len(checksums) and len(pending) < (
                        self._upload_concurrency
                    ):
                        pending.add(pool.submit(send, index))
                        index += 1
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        parts.append(future.result())
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        return sorted(parts, key=lambda part: part["PartNumber"])

    def _abort(self, key: str, upload_id: str) -> None:
        """Abandon an upload. One the store refuses to abort — a session that has
        ended with it — stays in the store."""
        with contextlib.suppress(Exception):
            self._client.abort_multipart_upload(
                Bucket=self._bucket, Key=key, UploadId=upload_id
            )

    def _present(self, reference: ContentReference) -> bool:
        """Whether the object is already stored, so a large write can skip sending it.

        A write that finds nothing still commits conditionally, so a concurrent writer
        landing first is caught at the commit. One that finds an object of another
        size fails at once, as its commit would after sending every part.
        """
        try:
            head = self._client.head_object(
                Bucket=self._bucket, Key=self._key(reference)
            )
        except Exception:
            return False
        check_existing_size(reference, int(head.get("ContentLength", -1)))
        return True

    def _check_existing(self, reference: ContentReference) -> None:
        """Take an object already under the reference's key as the one written, once
        its size shows it is not a truncated copy."""
        try:
            head = self._client.head_object(
                Bucket=self._bucket, Key=self._key(reference)
            )
        except Exception as exc:
            raise ContentStoreError(
                f"content {reference.content_digest} is neither written nor readable: "
                f"{exc}"
            ) from exc
        check_existing_size(reference, int(head.get("ContentLength", -1)))

    def stored_size(self, reference: ContentReference) -> int:
        try:
            head = self._client.head_object(
                Bucket=self._bucket, Key=self._key(reference)
            )
        except Exception as exc:
            raise self._read_error(reference, exc) from exc
        return int(head.get("ContentLength", -1))

    def fetch(self, reference: ContentReference) -> bytes:
        body = self._get(reference)
        try:
            return bytes(body.read())
        except Exception as exc:
            raise self._read_error(reference, exc) from exc
        finally:
            body.close()

    def open(
        self, reference: ContentReference, *, chunk_bytes: int = STREAM_CHUNK_BYTES
    ) -> Iterator[bytes]:
        body = self._get(reference)

        def chunks() -> Iterator[bytes]:
            try:
                yield from body.iter_chunks(chunk_bytes)
            except Exception as exc:
                raise self._read_error(reference, exc) from exc
            finally:
                body.close()

        return verified_chunks(reference, chunks())

    def open_range(
        self,
        reference: ContentReference,
        start: int,
        length: int,
        *,
        chunk_bytes: int = STREAM_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        check_range(reference, start, length)
        body = self._get(reference, byte_range=f"bytes={start}-{start + length - 1}")

        def chunks() -> Iterator[bytes]:
            got = 0
            try:
                for chunk in body.iter_chunks(chunk_bytes):
                    got += len(chunk)
                    if got > length:
                        raise ContentUnavailable(
                            f"a ranged read of {reference.content_digest} overran "
                            f"its {length} bytes"
                        )
                    yield chunk
            except ContentUnavailable:
                raise
            except Exception as exc:
                raise self._read_error(reference, exc) from exc
            finally:
                body.close()
            short_range(reference, got, length)

        return chunks()

    def _get(self, reference: ContentReference, byte_range: str | None = None) -> Any:
        extra = {"Range": byte_range} if byte_range is not None else {}
        try:
            response: dict[str, Any] = self._client.get_object(
                Bucket=self._bucket, Key=self._key(reference), **extra
            )
        except Exception as exc:
            raise self._read_error(reference, exc) from exc
        return response["Body"]

    def _read_error(
        self, reference: ContentReference, exc: Exception
    ) -> ContentHydrationError:
        if _is_missing(exc):
            return ContentHydrationError(
                f"no content for {reference.content_digest} in scope "
                f"{reference.authorization_scope}: {exc}"
            )
        return ContentUnavailable(
            f"could not read {reference.content_digest} in scope "
            f"{reference.authorization_scope}: {exc}"
        )


def _pread_exactly(fd: int, size: int, offset: int) -> bytes:
    """``size`` bytes of a file from ``offset``; one ``pread`` returns at most about
    2 GiB, so a larger part is read in pieces."""
    pieces: list[bytes] = []
    while size > 0:
        piece = os.pread(fd, size, offset)
        if not piece:
            break
        pieces.append(piece)
        size -= len(piece)
        offset += len(piece)
    return b"".join(pieces)


def _error_code(exc: Exception) -> tuple[str | None, int | None]:
    if not isinstance(exc, ClientError):
        return None, None
    error = exc.response.get("Error", {})
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return error.get("Code"), status


def _is_missing(exc: Exception) -> bool:
    """Whether a read failed because the store answered that the object is absent.

    Anything else — a timeout, a dropped connection, a throttle, a server error, a
    refused credential — leaves the object where it was, so the read may be retried.
    """
    code, status = _error_code(exc)
    return code in _MISSING_CODES or status == 404


def _committing[T](call: Callable[[], T]) -> T:
    """Make a conditional commit, again while it races another writer of its key."""
    attempt = 0
    while True:
        try:
            return call()
        except ClientError as exc:
            code, status = _error_code(exc)
            attempt += 1
            if not (code in _CONFLICT_CODES or status == 409) or (
                attempt == _CONFLICT_RETRIES
            ):
                raise
            time.sleep(_CONFLICT_BACKOFF_SEC * attempt)


def is_credential_refusal(exc: Exception) -> bool:
    """Whether the store refused a call for the session it was made under: one that ran
    out, was revoked, or does not cover the call."""
    return _error_code(exc)[0] in _CREDENTIAL_CODES


def _write_error(exc: Exception) -> ContentStoreError:
    code, status = _error_code(exc)
    if code == "NotImplemented" or status == 501:
        return ContentStoreError(
            "content write failed: the store does not support conditional writes "
            f"with SHA-256 checksums, which content storage requires: {exc}"
        )
    return ContentStoreError(f"content write failed: {exc}")


def _is_gone(exc: Exception) -> bool:
    """Whether a multipart call found its upload no longer there."""
    return _error_code(exc)[0] == "NoSuchUpload"


def _is_taken(exc: Exception) -> bool:
    """Whether a conditional write was refused because its key is already taken."""
    code, status = _error_code(exc)
    return code in _TAKEN_CODES or status == 412


__all__ = [
    "S3Client",
    "S3ObjectStore",
    "is_credential_refusal",
]

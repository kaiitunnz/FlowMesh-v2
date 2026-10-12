"""An in-memory S3 that enforces conditional writes and per-part checksums."""

import base64
import hashlib
import threading
from collections.abc import Iterator
from typing import Any

from botocore.exceptions import ClientError

from shared.content.config import MIN_PART_BYTES


def s3_error(code: str, status: int, op: str = "PutObject") -> ClientError:
    response: Any = {
        "Error": {"Code": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, op)


def sha256_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode()


def _etag(data: bytes) -> str:
    return f'"{hashlib.md5(data, usedforsecurity=False).hexdigest()}"'


class UnusedWrites:
    """The multipart calls an S3 client offers, for a fake whose test never makes
    them."""

    def head_object(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def create_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def upload_part(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def complete_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def abort_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")


class _Body:
    def __init__(self, data: bytes, *, fail_after: int | None = None) -> None:
        self._data = data
        self._fail_after = fail_after

    def read(self) -> bytes:
        if self._fail_after is not None and self._fail_after < len(self._data):
            raise ConnectionResetError("reset mid-body")
        return self._data

    def iter_chunks(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self._data), chunk_size):
            if self._fail_after is not None and start >= self._fail_after:
                raise ConnectionResetError("reset mid-body")
            yield self._data[start : start + chunk_size]

    def close(self) -> None:
        pass


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.aborted: list[str] = []
        self.fail_part: int | None = None
        # A store that ignores the checksums it is sent, as a writer must not rely on.
        self.check_checksums = True
        self.body_fail_after: int | None = None
        self._lock = threading.Lock()
        # Multipart uploads started, aborted ones included.
        self.created = 0

    def put_object(self, **kw: Any) -> Any:
        data = kw["Body"]
        if self.check_checksums and kw.get("ChecksumSHA256") not in (
            None,
            sha256_b64(data),
        ):
            raise s3_error("XAmzContentChecksumMismatch", 400)
        with self._lock:
            if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.objects:
                raise s3_error("PreconditionFailed", 412)
            self.objects[kw["Key"]] = bytes(data)
        return {}

    def get_object(self, **kw: Any) -> Any:
        if kw["Key"] not in self.objects:
            raise s3_error("NoSuchKey", 404, "GetObject")
        return {"Body": _Body(self.objects[kw["Key"]], fail_after=self.body_fail_after)}

    def head_object(self, **kw: Any) -> Any:
        if kw["Key"] not in self.objects:
            raise s3_error("404", 404, "HeadObject")
        return {"ContentLength": len(self.objects[kw["Key"]])}

    def create_multipart_upload(self, **kw: Any) -> Any:
        if kw.get("ChecksumAlgorithm") != "SHA256":
            raise s3_error("InvalidRequest", 400, "CreateMultipartUpload")
        with self._lock:
            self.created += 1
            upload_id = f"up-{self.created}"
            self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, **kw: Any) -> Any:
        data = kw["Body"]
        if self.fail_part is not None and kw["PartNumber"] == self.fail_part:
            raise s3_error("InternalError", 500, "UploadPart")
        if kw.get("ChecksumAlgorithm") != "SHA256":
            raise s3_error("InvalidRequest", 400, "UploadPart")
        if self.check_checksums and kw["ChecksumSHA256"] != sha256_b64(data):
            raise s3_error("XAmzContentChecksumMismatch", 400, "UploadPart")
        with self._lock:
            self.uploads[kw["UploadId"]][kw["PartNumber"]] = bytes(data)
        return {"ETag": _etag(data)}

    def complete_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            if kw["UploadId"] not in self.uploads:
                raise s3_error("NoSuchUpload", 404, "CompleteMultipartUpload")
            uploaded = self.uploads[kw["UploadId"]]
            if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.objects:
                raise s3_error("PreconditionFailed", 412, "CompleteMultipartUpload")
            listed = kw["MultipartUpload"]["Parts"]
            numbers = [part["PartNumber"] for part in listed]
            if numbers != sorted(set(numbers)):
                raise s3_error("InvalidPartOrder", 400, "CompleteMultipartUpload")
            for index, part in enumerate(listed):
                data = uploaded.get(part["PartNumber"])
                if (
                    data is None
                    or part["ETag"] != _etag(data)
                    or part.get("ChecksumSHA256") != sha256_b64(data)
                ):
                    raise s3_error("InvalidPart", 400, "CompleteMultipartUpload")
                if index < len(listed) - 1 and len(data) < MIN_PART_BYTES:
                    raise s3_error("EntityTooSmall", 400, "CompleteMultipartUpload")
            self.objects[kw["Key"]] = b"".join(uploaded[n] for n in numbers)
            del self.uploads[kw["UploadId"]]
        return {}

    def abort_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            self.uploads.pop(kw["UploadId"], None)
            self.aborted.append(kw["UploadId"])
        return {}

"""An in-memory S3 that enforces conditional writes and per-part checksums."""

import base64
import hashlib
import threading
from collections.abc import Iterator
from typing import Any

from botocore.exceptions import ClientError


def s3_error(code: str, status: int, op: str = "PutObject") -> ClientError:
    response: Any = {
        "Error": {"Code": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, op)


def sha256_b64(data: bytes) -> str:
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
        if kw.get("ChecksumSHA256") not in (None, sha256_b64(data)):
            raise s3_error("XAmzContentChecksumMismatch", 400)
        with self._lock:
            self.puts += 1
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
        with self._lock:
            self._next += 1
            upload_id = f"up-{self._next}"
            self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, **kw: Any) -> Any:
        data = kw["Body"]
        if self.fail_part is not None and kw["PartNumber"] == self.fail_part:
            raise s3_error("InternalError", 500, "UploadPart")
        if kw["ChecksumSHA256"] != sha256_b64(data):
            raise s3_error("XAmzContentChecksumMismatch", 400, "UploadPart")
        with self._lock:
            self.uploads[kw["UploadId"]][kw["PartNumber"]] = bytes(data)
        return {"ETag": f"etag-{kw['PartNumber']}"}

    def complete_multipart_upload(self, **kw: Any) -> Any:
        with self._lock:
            if kw["UploadId"] not in self.uploads:
                raise s3_error("NoSuchUpload", 404, "CompleteMultipartUpload")
            parts = self.uploads[kw["UploadId"]]
            if kw.get("IfNoneMatch") == "*" and kw["Key"] in self.objects:
                raise s3_error("PreconditionFailed", 412, "CompleteMultipartUpload")
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

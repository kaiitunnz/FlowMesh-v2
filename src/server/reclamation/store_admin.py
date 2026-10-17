"""Bucket-level reads the collector makes of an S3-compatible content store.

These run under the deployment's own store credential, which can list the bucket's
object versions and multipart uploads and read its versioning state; the scoped
sessions workers hold can do none of that. Nothing here writes, deletes or aborts.

A MinIO-family store, the co-located one included, lists open multipart uploads only
by exact key and from memory a restart empties, so a listing of its open uploads can
omit uploads still open. Asked about one UploadId, every store answers exactly.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from shared.content import BACKEND_S3, DigestAlgorithm, ObjectStoreConfig

# The most a listing returns per page.
PAGE_SIZE = 1000

_DIGEST = re.compile(r"[0-9a-f]{64}")


def admin_client(cfg: ObjectStoreConfig) -> Any:
    """An S3 client under the deployment credential, with short timeouts and few
    attempts so an unreachable store costs a sweep a moment."""
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url or None,
        aws_access_key_id=cfg.access_key or None,
        aws_secret_access_key=cfg.secret_key or None,
        region_name=cfg.region,
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"total_max_attempts": 2},
        ),
    )


class Ineligibility(StrEnum):
    """Why a store is not collected."""

    FILESYSTEM = "filesystem_backend"
    """A shared filesystem offers no versioned, conditional delete."""
    UNVERSIONED = "versioning_not_enabled"
    VERSIONING_UNKNOWN = "versioning_unreadable"
    CANNOT_LIST = "cannot_list"


@dataclass(frozen=True)
class Eligibility:
    """Whether the store may be collected, and why not when it may not."""

    reason: Ineligibility | None = None
    detail: str = ""

    @property
    def eligible(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class StoredVersion:
    """One version of one key under a scope."""

    key: str
    version_id: str
    size_bytes: int
    last_modified: float
    latest: bool
    digest: str | None
    """The content digest the key names, or None for a key that names no content
    object."""


@dataclass(frozen=True)
class OpenUpload:
    """A multipart upload a listing names open."""

    key: str
    upload_id: str
    initiated: float
    scope: str | None


def _timestamp(value: Any) -> float:
    return value.timestamp() if isinstance(value, datetime) else 0.0


class StoreAdmin:
    """Lists one bucket's content under the deployment's key prefix."""

    def __init__(self, client: Any, bucket: str, prefix: str = "") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    @classmethod
    def from_config(cls, cfg: ObjectStoreConfig) -> "StoreAdmin | None":
        """The admin view of an S3-compatible store; None for a filesystem store."""
        if cfg.backend != BACKEND_S3:
            return None
        return cls(admin_client(cfg), cfg.bucket, cfg.prefix)

    def _root(self, scope: str = "") -> str:
        parts = [part for part in (self._prefix, scope) if part]
        return "/".join(parts) + "/" if parts else ""

    def eligibility(self) -> Eligibility:
        """Whether the bucket keeps versions and the credential can list it."""
        try:
            status = self._client.get_bucket_versioning(Bucket=self._bucket).get(
                "Status"
            )
        except Exception as exc:
            return Eligibility(Ineligibility.VERSIONING_UNKNOWN, str(exc))
        if status != "Enabled":
            return Eligibility(
                Ineligibility.UNVERSIONED, str(status or "never enabled")
            )
        try:
            self._client.list_object_versions(
                Bucket=self._bucket, Prefix=self._root(), MaxKeys=1
            )
            self._client.list_multipart_uploads(
                Bucket=self._bucket, Prefix=self._root(), MaxUploads=1
            )
        except Exception as exc:
            return Eligibility(Ineligibility.CANNOT_LIST, str(exc))
        return Eligibility()

    def scopes(self) -> Iterator[str]:
        """Every scope the bucket holds a key under."""
        root = self._root()
        token: str | None = None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            page = self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=root,
                Delimiter="/",
                MaxKeys=PAGE_SIZE,
                **extra,
            )
            for common in page.get("CommonPrefixes", []):
                if scope := common["Prefix"][len(root) :].rstrip("/"):
                    yield scope
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")

    def versions(self, scope: str) -> Iterator[list[StoredVersion]]:
        """Every object version under ``scope``, a page at a time."""
        root = self._root(scope)
        content = f"{root}{DigestAlgorithm.SHA256.value}/"
        markers: dict[str, str] = {}
        while True:
            page = self._client.list_object_versions(
                Bucket=self._bucket, Prefix=root, MaxKeys=PAGE_SIZE, **markers
            )
            listed: list[StoredVersion] = []
            for version in page.get("Versions", []):
                key = version["Key"]
                name = key[len(content) :] if key.startswith(content) else ""
                listed.append(
                    StoredVersion(
                        key=key,
                        version_id=str(version.get("VersionId") or "null"),
                        size_bytes=int(version.get("Size") or 0),
                        last_modified=_timestamp(version.get("LastModified")),
                        latest=bool(version.get("IsLatest")),
                        digest=name if _DIGEST.fullmatch(name) else None,
                    )
                )
            yield listed
            if not page.get("IsTruncated"):
                return
            markers = {
                "KeyMarker": page.get("NextKeyMarker", ""),
                "VersionIdMarker": page.get("NextVersionIdMarker", ""),
            }

    def uploads(self) -> Iterator[list[OpenUpload]]:
        """Every multipart upload the bucket holds open under the prefix, a page at a
        time."""
        root = self._root()
        markers: dict[str, str] = {}
        while True:
            page = self._client.list_multipart_uploads(
                Bucket=self._bucket, Prefix=root, MaxUploads=PAGE_SIZE, **markers
            )
            listed: list[OpenUpload] = []
            for upload in page.get("Uploads", []):
                key = upload["Key"]
                scope, sep, _ = key[len(root) :].partition("/")
                listed.append(
                    OpenUpload(
                        key=key,
                        upload_id=upload["UploadId"],
                        initiated=_timestamp(upload.get("Initiated")),
                        scope=scope if sep else None,
                    )
                )
            yield listed
            if not page.get("IsTruncated"):
                return
            markers = {
                "KeyMarker": page.get("NextKeyMarker", ""),
                "UploadIdMarker": page.get("NextUploadIdMarker", ""),
            }

    def upload_open(self, key: str, upload_id: str) -> bool:
        """Whether the store still holds the upload ``upload_id`` of ``key`` open.

        Raises when the store gives no answer.
        """
        try:
            self._client.list_parts(
                Bucket=self._bucket, Key=key, UploadId=upload_id, MaxParts=1
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchUpload":
                return False
            raise
        return True

    def upload_bytes(self, upload: OpenUpload) -> tuple[int, int]:
        """The parts an open upload holds and their total size."""
        parts = size = 0
        marker = 0
        while True:
            page = self._client.list_parts(
                Bucket=self._bucket,
                Key=upload.key,
                UploadId=upload.upload_id,
                MaxParts=PAGE_SIZE,
                PartNumberMarker=marker,
            )
            for part in page.get("Parts", []):
                parts += 1
                size += int(part.get("Size") or 0)
            if not page.get("IsTruncated"):
                return parts, size
            marker = int(page.get("NextPartNumberMarker") or 0)


__all__ = [
    "Eligibility",
    "Ineligibility",
    "OpenUpload",
    "PAGE_SIZE",
    "StoreAdmin",
    "StoredVersion",
    "admin_client",
]

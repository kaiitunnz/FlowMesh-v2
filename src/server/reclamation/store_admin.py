"""Bucket-level reads the audit makes of an S3-compatible content store.

These run under the deployment's own store credential, which can list the bucket's
objects, their versions and its multipart uploads and read its versioning state; the
scoped sessions workers hold can do none of that. Nothing here writes, deletes or
aborts.

Only AWS S3 is known to list every open multipart upload of a bucket; a MinIO-family
store, the co-located one included, lists them only by exact key and from memory a
restart empties. A listing of open uploads counts as complete only on AWS S3, and only
bucket-wide.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Self
from urllib.parse import urlsplit

import boto3
from botocore.client import Config

from shared.content import BACKEND_S3, DigestAlgorithm, ObjectStoreConfig

from .pacing import Pacer

# The most a listing returns per page, kept small since parsing a page holds the
# interpreter.
PAGE_SIZE = 250

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


def uploads_listing_complete(cfg: ObjectStoreConfig) -> bool:
    """Whether a listing of the store's open multipart uploads names every one: on AWS
    S3, listed bucket-wide."""
    if cfg.backend != BACKEND_S3 or cfg.prefix.strip("/"):
        return False
    if not cfg.endpoint_url:
        return True
    host = urlsplit(cfg.endpoint_url).hostname or ""
    return host == "amazonaws.com" or host.endswith(".amazonaws.com")


class Ineligibility(StrEnum):
    """Why a store is not audited."""

    FILESYSTEM = "filesystem_backend"
    CANNOT_LIST = "cannot_list"


class Listing(StrEnum):
    """What an audit lists of an eligible store."""

    VERSIONS = "versions"
    """Every version of every object, the bucket keeping or having kept versions."""
    OBJECTS = "objects"
    """Every object, the bucket never having kept versions."""


@dataclass(frozen=True)
class Eligible:
    """A store the audit may list, and what it lists of it."""

    listing: Listing
    versioning: str
    """The bucket's versioning state as the store reports it."""


@dataclass(frozen=True)
class Ineligible:
    """A store the audit cannot list, and why."""

    reason: Ineligibility
    detail: str = ""


type Eligibility = Eligible | Ineligible


class ListingError(Exception):
    """The store answered a listing it cannot be paged through."""


@dataclass(frozen=True)
class StoredVersion:
    """One version of one key under a scope."""

    size_bytes: int
    last_modified: float
    digest: str | None
    """The content digest the key names, or None for a key that names no content
    object."""


@dataclass(frozen=True)
class OpenUpload:
    """A multipart upload a listing names open."""

    key: str
    upload_id: str
    scope: str | None


def _timestamp(value: Any) -> float:
    return value.timestamp() if isinstance(value, datetime) else 0.0


def _next(page: dict[str, Any], *names: str) -> dict[str, str]:
    """The markers continuing a truncated listing past ``page``."""
    markers = {name: str(value) for name in names if (value := page.get(f"Next{name}"))}
    if names[0] not in markers:
        raise ListingError(
            f"the store reported a truncated listing without a Next{names[0]}"
        )
    return markers


class StoreAdmin:
    """Lists one bucket's content under the deployment's key prefix."""

    def __init__(self, client: Any, bucket: str, prefix: str = "") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    @classmethod
    def from_config(cls, cfg: ObjectStoreConfig) -> Self | None:
        """The admin view of an S3-compatible store; None for a filesystem store."""
        if cfg.backend != BACKEND_S3:
            return None
        return cls(admin_client(cfg), cfg.bucket, cfg.prefix)

    def _root(self, scope: str = "") -> str:
        parts = [part for part in (self._prefix, scope) if part]
        return "/".join(parts) + "/" if parts else ""

    def eligibility(self) -> Eligibility:
        """What the credential can list of the bucket: its versions where it keeps or
        kept them, or may have, else its objects."""
        try:
            status = str(
                self._client.get_bucket_versioning(Bucket=self._bucket).get("Status")
                or "off"
            )
        except Exception as exc:
            status = f"unreadable: {exc}"
        try:
            listing = self._probe(status)
            self._client.list_multipart_uploads(
                Bucket=self._bucket, Prefix=self._root(), MaxUploads=1
            )
        except Exception as exc:
            return Ineligible(Ineligibility.CANNOT_LIST, str(exc))
        return Eligible(listing, status)

    def _probe(self, status: str) -> Listing:
        """The listing the credential can make for a bucket of versioning ``status``;
        a bucket of unreadable status is listed by version where that is allowed."""
        if status == "off":
            self._client.list_objects_v2(
                Bucket=self._bucket, Prefix=self._root(), MaxKeys=1
            )
            return Listing.OBJECTS
        try:
            self._client.list_object_versions(
                Bucket=self._bucket, Prefix=self._root(), MaxKeys=1
            )
        except Exception:
            if status in ("Enabled", "Suspended"):
                raise
            return self._probe("off")
        return Listing.VERSIONS

    def listed(self, listing: Listing, scope: str) -> Iterator[list[StoredVersion]]:
        """Every object, or every object version, under ``scope``, a page at a time."""
        if listing is Listing.VERSIONS:
            return self.versions(scope)
        return self.objects(scope)

    def scopes(self, listing: Listing) -> Iterator[list[str]]:
        """Every scope the bucket holds a key, or a key version, under, a page at a
        time."""
        root = self._root()
        markers: dict[str, str] = {}
        while True:
            if listing is Listing.VERSIONS:
                page = self._client.list_object_versions(
                    Bucket=self._bucket,
                    Prefix=root,
                    Delimiter="/",
                    MaxKeys=PAGE_SIZE,
                    **markers,
                )
            else:
                page = self._client.list_objects_v2(
                    Bucket=self._bucket,
                    Prefix=root,
                    Delimiter="/",
                    MaxKeys=PAGE_SIZE,
                    **markers,
                )
            yield [
                scope
                for common in page.get("CommonPrefixes", [])
                if (scope := common["Prefix"][len(root) :].rstrip("/"))
            ]
            if not page.get("IsTruncated"):
                return
            if listing is Listing.VERSIONS:
                markers = _next(page, "KeyMarker", "VersionIdMarker")
            else:
                markers = _next(page, "ContinuationToken")

    def versions(self, scope: str) -> Iterator[list[StoredVersion]]:
        """Every object version under ``scope``, a page at a time."""
        root = self._root(scope)
        content = f"{root}{DigestAlgorithm.SHA256.value}/"
        markers: dict[str, str] = {}
        while True:
            page = self._client.list_object_versions(
                Bucket=self._bucket, Prefix=root, MaxKeys=PAGE_SIZE, **markers
            )
            yield [_stored(version, content) for version in page.get("Versions", [])]
            if not page.get("IsTruncated"):
                return
            markers = _next(page, "KeyMarker", "VersionIdMarker")

    def objects(self, scope: str) -> Iterator[list[StoredVersion]]:
        """Every object under ``scope``, each its one version, a page at a time."""
        root = self._root(scope)
        content = f"{root}{DigestAlgorithm.SHA256.value}/"
        markers: dict[str, str] = {}
        while True:
            page = self._client.list_objects_v2(
                Bucket=self._bucket, Prefix=root, MaxKeys=PAGE_SIZE, **markers
            )
            yield [_stored(item, content) for item in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return
            markers = _next(page, "ContinuationToken")

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
                    OpenUpload(key, upload["UploadId"], scope if sep else None)
                )
            yield listed
            if not page.get("IsTruncated"):
                return
            markers = _next(page, "KeyMarker", "UploadIdMarker")

    def upload_bytes(self, upload: OpenUpload, pacer: Pacer) -> tuple[int, int]:
        """The parts an open upload holds and their total size, ticking ``pacer`` per
        page; a stopped pacer ends it with what it has counted."""
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
            if not page.get("IsTruncated") or pacer.tick():
                return parts, size
            marker = int(_next(page, "PartNumberMarker")["PartNumberMarker"])


def _stored(item: dict[str, Any], content: str) -> StoredVersion:
    key = item["Key"]
    name = key[len(content) :] if key.startswith(content) else ""
    return StoredVersion(
        size_bytes=int(item.get("Size") or 0),
        last_modified=_timestamp(item.get("LastModified")),
        digest=name if _DIGEST.fullmatch(name) else None,
    )


__all__ = [
    "Eligibility",
    "Eligible",
    "Ineligibility",
    "Ineligible",
    "Listing",
    "ListingError",
    "OpenUpload",
    "PAGE_SIZE",
    "StoreAdmin",
    "StoredVersion",
    "admin_client",
    "uploads_listing_complete",
]

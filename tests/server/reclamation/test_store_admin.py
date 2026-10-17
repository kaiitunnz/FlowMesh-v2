"""The store admin: what an audit may list of a bucket, its paged listings, and when a
listing of open uploads is complete; it only ever reads."""

import logging
import threading
from datetime import UTC, datetime
from typing import Any

import fakeredis
import pytest

from server.reclamation.audit import AuditTrail
from server.reclamation.closures import TreeClosures
from server.reclamation.collector import AuditCollector
from server.reclamation.marking import Marker
from server.reclamation.pacing import Pacer
from server.reclamation.service import ContentAuditService
from server.reclamation.store_admin import (
    Eligible,
    Ineligibility,
    Ineligible,
    Listing,
    ListingError,
    StoreAdmin,
    uploads_listing_complete,
)
from shared.content import (
    BACKEND_FILESYSTEM,
    BACKEND_S3,
    ObjectStoreConfig,
    content_digest,
)
from tests.server.redis_helpers import fake_redis_client

PREFIX = "fabric"
NOW = 1_000_000.0
OLD = NOW - 7200
_LOGGER = logging.getLogger("test.store_admin")

READS = frozenset(
    {
        "get_bucket_versioning",
        "list_objects_v2",
        "list_object_versions",
        "list_multipart_uploads",
        "list_parts",
    }
)


def when(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


class ReadOnlyS3:
    """An S3 client over an in-memory bucket offering only the reads an audit makes,
    so any write, delete or abort raises AttributeError."""

    def __init__(self, versioning: str | None = "Enabled", page: int = 1000) -> None:
        self.versioning = versioning
        self.page = page
        self.untokened = False
        """Answer a truncated page with no continuation, as a broken store might."""
        self.items: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.parts: dict[str, list[dict[str, Any]]] = {}
        self.fail: set[str] = set()
        self.calls: list[str] = []

    def put(
        self,
        scope: str,
        data: bytes,
        *,
        at: float = OLD,
        version: str = "v1",
        latest: bool = True,
    ) -> str:
        digest = content_digest(data)
        self.put_key(
            f"{PREFIX}/{scope}/sha256/{digest}", len(data), at, version, latest
        )
        return digest

    def put_key(
        self,
        key: str,
        size: int = 1,
        at: float = OLD,
        version: str = "v1",
        latest: bool = True,
    ) -> None:
        self.items.append(
            {
                "Key": key,
                "VersionId": version,
                "Size": size,
                "LastModified": when(at),
                "IsLatest": latest,
            }
        )
        self.items.sort(key=lambda item: (item["Key"], item["VersionId"]))

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise PermissionError("AccessDenied")

    def _continued(self, out: dict[str, Any], **markers: Any) -> dict[str, Any]:
        if out["IsTruncated"] and not self.untokened:
            out.update(markers)
        return out

    @staticmethod
    def _delimited(prefix: str, keys: list[str], delimiter: str) -> list[str]:
        return sorted(
            {
                f"{prefix}{key[len(prefix):].split(delimiter)[0]}{delimiter}"
                for key in keys
            }
        )

    def get_bucket_versioning(self, Bucket: str) -> dict[str, Any]:
        self._call("get_bucket_versioning")
        return {"Status": self.versioning} if self.versioning else {}

    def list_objects_v2(
        self,
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
        Delimiter: str = "",
        ContinuationToken: str = "",
    ) -> dict[str, Any]:
        self._call("list_objects_v2")
        latest = {item["Key"]: item for item in self.items if item["IsLatest"]}
        keys = sorted(key for key in latest if key.startswith(Prefix))
        entries = self._delimited(Prefix, keys, Delimiter) if Delimiter else keys
        start = int(ContinuationToken or 0)
        size = min(MaxKeys, self.page)
        page = entries[start : start + size]
        out: dict[str, Any] = {"IsTruncated": start + size < len(entries)}
        if Delimiter:
            out["CommonPrefixes"] = [{"Prefix": prefix} for prefix in page]
        else:
            out["Contents"] = [
                {k: latest[key][k] for k in ("Key", "Size", "LastModified")}
                for key in page
            ]
        return self._continued(out, NextContinuationToken=str(start + size))

    def list_object_versions(
        self,
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
        Delimiter: str = "",
        KeyMarker: str = "",
        VersionIdMarker: str = "",
    ) -> dict[str, Any]:
        self._call("list_object_versions")
        size = min(MaxKeys, self.page)
        if Delimiter:
            keys = [
                item["Key"] for item in self.items if item["Key"].startswith(Prefix)
            ]
            prefixes = [
                p for p in self._delimited(Prefix, keys, Delimiter) if p > KeyMarker
            ]
            out = {
                "CommonPrefixes": [{"Prefix": p} for p in prefixes[:size]],
                "IsTruncated": len(prefixes) > size,
            }
            return self._continued(out, NextKeyMarker="".join(prefixes[:size][-1:]))
        listed = [
            item
            for item in self.items
            if item["Key"].startswith(Prefix)
            and (item["Key"], item["VersionId"]) > (KeyMarker, VersionIdMarker)
        ]
        page = listed[:size]
        out = {"Versions": page, "IsTruncated": len(listed) > size}
        return self._continued(
            out,
            NextKeyMarker=page[-1]["Key"] if page else "",
            NextVersionIdMarker=page[-1]["VersionId"] if page else "",
        )

    def list_multipart_uploads(
        self,
        Bucket: str,
        Prefix: str,
        MaxUploads: int,
        KeyMarker: str = "",
        UploadIdMarker: str = "",
    ) -> dict[str, Any]:
        self._call("list_multipart_uploads")
        listed = [u for u in self.uploads if u["Key"].startswith(Prefix)]
        return {"Uploads": listed[:MaxUploads], "IsTruncated": False}

    def list_parts(
        self, Bucket: str, Key: str, UploadId: str, MaxParts: int, PartNumberMarker: int
    ) -> dict[str, Any]:
        self._call("list_parts")
        parts = self.parts.get(UploadId, [])[PartNumberMarker:]
        size = min(MaxParts, self.page)
        out: dict[str, Any] = {"Parts": parts[:size], "IsTruncated": len(parts) > size}
        if out["IsTruncated"]:
            out["NextPartNumberMarker"] = PartNumberMarker + size
        return out


def _admin(s3: ReadOnlyS3) -> StoreAdmin:
    return StoreAdmin(s3, "bucket", PREFIX)


@pytest.mark.parametrize("status", ["Enabled", "Suspended"])
def test_a_bucket_that_keeps_or_kept_versions_is_listed_by_version(status: str) -> None:
    s3 = ReadOnlyS3(status)

    eligibility = _admin(s3).eligibility()

    assert eligibility == Eligible(Listing.VERSIONS, status)
    assert "list_objects_v2" not in s3.calls


def test_a_bucket_that_never_kept_versions_is_listed_by_object() -> None:
    s3 = ReadOnlyS3(None)

    eligibility = _admin(s3).eligibility()

    assert eligibility == Eligible(Listing.OBJECTS, "off")
    assert "list_object_versions" not in s3.calls


@pytest.mark.parametrize("versions_refused", [False, True])
def test_a_bucket_of_unreadable_versioning_is_listed_by_version_where_allowed(
    versions_refused: bool,
) -> None:
    s3 = ReadOnlyS3()
    s3.fail.add("get_bucket_versioning")
    if versions_refused:
        s3.fail.add("list_object_versions")

    eligibility = _admin(s3).eligibility()

    assert isinstance(eligibility, Eligible)
    assert eligibility.listing is (
        Listing.OBJECTS if versions_refused else Listing.VERSIONS
    )
    assert eligibility.versioning.startswith("unreadable: ")


@pytest.mark.parametrize(
    ("status", "refused"),
    [
        ("Enabled", "list_object_versions"),
        ("Suspended", "list_object_versions"),
        (None, "list_objects_v2"),
        ("Enabled", "list_multipart_uploads"),
    ],
)
def test_a_credential_that_cannot_list_is_ineligible(
    status: str | None, refused: str
) -> None:
    s3 = ReadOnlyS3(status)
    s3.fail.add(refused)

    eligibility = _admin(s3).eligibility()

    assert isinstance(eligibility, Ineligible)
    assert eligibility.reason is Ineligibility.CANNOT_LIST
    assert "AccessDenied" in eligibility.detail


@pytest.mark.parametrize("listing", list(Listing))
def test_a_listing_pages_every_key_and_names_only_content_digests(
    listing: Listing,
) -> None:
    s3 = ReadOnlyS3(page=2)
    digests = {s3.put("org-a", f"object-{i}".encode()) for i in range(5)}
    s3.put_key(f"{PREFIX}/org-a/staging/leftover")
    s3.put_key(f"{PREFIX}/org-a/sha256/not-a-digest")
    s3.put("org-b", b"elsewhere")

    pages = list(_admin(s3).listed(listing, "org-a"))

    listed = [version for page in pages for version in page]
    assert len(pages) == 4 and all(len(page) <= 2 for page in pages)
    assert {v.digest for v in listed if v.digest} == digests
    assert len([v for v in listed if v.digest is None]) == 2
    assert all(v.last_modified == OLD for v in listed)


def test_versions_lists_every_version_of_a_key() -> None:
    s3 = ReadOnlyS3(page=1)
    s3.put("org-a", b"same", version="v1")
    s3.put("org-a", b"same", version="v2")

    listed = [v for page in _admin(s3).versions("org-a") for v in page]

    assert len(listed) == 2 and len({v.digest for v in listed}) == 1


def test_objects_lists_one_version_per_key() -> None:
    s3 = ReadOnlyS3(None)
    s3.put("org-a", b"same", version="v1")
    s3.put("org-a", b"same", version="v2")

    (listed,) = list(_admin(s3).objects("org-a"))

    assert len(listed) == 1


@pytest.mark.parametrize("listing", list(Listing))
def test_scopes_pages_every_scope_under_the_prefix(listing: Listing) -> None:
    s3 = ReadOnlyS3(page=1)
    for scope in ("org-c", "org-a", "org-b"):
        s3.put(scope, scope.encode())
    s3.put_key("elsewhere/org-z/sha256/x")

    pages = list(_admin(s3).scopes(listing))

    assert pages == [["org-a"], ["org-b"], ["org-c"]]
    assert ("list_object_versions" in s3.calls) is (listing is Listing.VERSIONS)


def test_a_scope_holding_only_noncurrent_versions_is_found_by_version() -> None:
    s3 = ReadOnlyS3("Suspended")
    s3.put("org-a", b"current")
    s3.put("org-old", b"deleted", latest=False)

    admin = _admin(s3)

    assert [s for page in admin.scopes(Listing.OBJECTS) for s in page] == ["org-a"]
    assert [s for page in admin.scopes(Listing.VERSIONS) for s in page] == [
        "org-a",
        "org-old",
    ]


@pytest.mark.parametrize("listing", list(Listing))
def test_a_truncated_page_with_no_continuation_stops_the_listing(
    listing: Listing,
) -> None:
    s3 = ReadOnlyS3(page=1)
    for i in range(3):
        s3.put("org-a", f"object-{i}".encode())
    s3.untokened = True
    listed = _admin(s3).listed(listing, "org-a")

    assert len(next(listed)) == 1
    with pytest.raises(ListingError):
        next(listed)


def test_uploads_name_their_scope_and_are_sized_by_their_parts() -> None:
    s3 = ReadOnlyS3(page=2)
    s3.uploads = [
        {"Key": f"{PREFIX}/org-a/sha256/x", "UploadId": "u-1", "Initiated": when(OLD)},
        {"Key": f"{PREFIX}/stray", "UploadId": "u-2", "Initiated": when(OLD)},
    ]
    s3.parts = {"u-1": [{"Size": 5}, {"Size": 6}, {"Size": 7}]}
    admin = _admin(s3)

    (uploads,) = list(admin.uploads())

    assert [(u.upload_id, u.scope) for u in uploads] == [
        ("u-1", "org-a"),
        ("u-2", None),
    ]
    assert admin.upload_bytes(uploads[0], Pacer(threading.Event(), 1, duty=1.0)) == (
        3,
        18,
    )


def test_sizing_an_upload_ends_at_the_page_its_pacer_stops() -> None:
    s3 = ReadOnlyS3(page=1)
    s3.uploads = [{"Key": f"{PREFIX}/org-a/sha256/x", "UploadId": "u-1"}]
    s3.parts = {"u-1": [{"Size": 5}, {"Size": 6}, {"Size": 7}]}
    admin = _admin(s3)
    (uploads,) = list(admin.uploads())
    stop = threading.Event()
    stop.set()

    assert admin.upload_bytes(uploads[0], Pacer(stop, 1, duty=1.0)) == (1, 5)
    assert s3.calls.count("list_parts") == 1


@pytest.mark.parametrize(
    ("backend", "endpoint", "prefix", "complete"),
    [
        (BACKEND_S3, "", "", True),
        (BACKEND_S3, "https://s3.amazonaws.com", "", True),
        (BACKEND_S3, "https://s3.eu-west-1.amazonaws.com", "/", True),
        (BACKEND_S3, "http://127.0.0.1:9000", "", False),
        (BACKEND_S3, "https://notamazonaws.com", "", False),
        (BACKEND_S3, "", "fabric", False),
        (BACKEND_S3, "https://s3.amazonaws.com", "fabric/", False),
        (BACKEND_FILESYSTEM, "", "", False),
    ],
)
def test_only_a_bucket_wide_aws_listing_of_uploads_is_complete(
    backend: str, endpoint: str, prefix: str, complete: bool
) -> None:
    cfg = ObjectStoreConfig(backend=backend, endpoint_url=endpoint, prefix=prefix)

    assert uploads_listing_complete(cfg) is complete


def test_a_full_sweep_only_reads_the_store() -> None:
    rds = fake_redis_client(fakeredis.FakeServer())
    s3 = ReadOnlyS3(page=1)
    s3.put("org-a", b"unbound")
    s3.put_key(f"{PREFIX}/org-a/staging/leftover")
    s3.uploads = [
        {"Key": f"{PREFIX}/org-a/sha256/x", "UploadId": "u-1", "Initiated": when(OLD)}
    ]
    s3.parts = {"u-1": [{"Size": 4}]}
    service = ContentAuditService(
        Marker(rds, _LOGGER, min_pause_sec=0, duty=1.0),
        TreeClosures(lambda ref: iter(()), _LOGGER),
        AuditCollector(
            _admin(s3),
            _LOGGER,
            AuditTrail(rds),
            recent_sec=0,
            duty=1.0,
            min_pause_sec=0,
        ),
        _LOGGER,
        duty=1.0,
        min_pause_sec=0,
    )

    result = service.sweep(threading.Event())

    assert isinstance(result.eligibility, Eligible) and result.unbound == 1
    assert result.uploads is not None
    assert (result.uploads.uploads, result.uploads.bytes) == (1, 4)
    assert result.uploads.unsized == 0
    assert set(s3.calls) <= READS and "list_parts" in s3.calls

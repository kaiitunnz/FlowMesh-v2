"""The audit collector: it condemns exactly the stored objects nothing binds or holds,
on an eligible store, and deletes and aborts nothing."""

import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_AUDIT_KEY,
    RECLAIM_SCOPES_KEY,
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)
from server.reclamation.audit import AuditTrail
from server.reclamation.backfill import FAMILIES
from server.reclamation.certification import CoverageStore
from server.reclamation.collector import AuditCollector
from server.reclamation.store_admin import Ineligibility, StoreAdmin
from shared.content import BACKEND_FILESYSTEM, ObjectStoreConfig, content_digest
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_PREFIX = "fabric"
_NOW = 1_000_000.0
_OLD = _NOW - 7200
_LOGGER = logging.getLogger("test.collector")


def _digest(name: str) -> str:
    return content_digest(name.encode())


def _when(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


class StubS3:
    """An S3 client over an in-memory listing that records every call and refuses
    any that would delete or abort."""

    def __init__(self, versioning: str | None = "Enabled", page: int = 1000) -> None:
        self.versioning = versioning
        self.page = page
        self.versions: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.parts: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the collector called {name}")

    def put(self, scope: str, name: str, *, size: int = 10, at: float = _OLD) -> str:
        digest = _digest(name)
        self.versions.append(
            {
                "Key": f"{_PREFIX}/{scope}/sha256/{digest}",
                "VersionId": f"v-{name}",
                "Size": size,
                "LastModified": _when(at),
                "IsLatest": True,
            }
        )
        self.versions.sort(key=lambda v: v["Key"])
        return digest

    def get_bucket_versioning(self, Bucket: str) -> dict[str, Any]:
        self.calls.append("get_bucket_versioning")
        return {"Status": self.versioning} if self.versioning else {}

    def list_object_versions(
        self,
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
        KeyMarker: str = "",
        VersionIdMarker: str = "",
    ) -> dict[str, Any]:
        self.calls.append("list_object_versions")
        listed = [
            v
            for v in self.versions
            if v["Key"].startswith(Prefix) and v["Key"] > KeyMarker
        ]
        size = min(MaxKeys, self.page)
        page = listed[:size]
        out: dict[str, Any] = {"Versions": page, "IsTruncated": len(listed) > size}
        if out["IsTruncated"]:
            out["NextKeyMarker"] = page[-1]["Key"]
            out["NextVersionIdMarker"] = page[-1]["VersionId"]
        return out

    def list_objects_v2(
        self, Bucket: str, Prefix: str, Delimiter: str, MaxKeys: int, **_: Any
    ) -> dict[str, Any]:
        self.calls.append("list_objects_v2")
        scopes = sorted(
            {
                v["Key"][len(Prefix) :].split(Delimiter)[0]
                for v in self.versions
                if v["Key"].startswith(Prefix)
            }
        )
        return {
            "CommonPrefixes": [{"Prefix": f"{Prefix}{s}/"} for s in scopes],
            "IsTruncated": False,
        }

    def list_multipart_uploads(
        self, Bucket: str, Prefix: str, MaxUploads: int, **_: Any
    ) -> dict[str, Any]:
        self.calls.append("list_multipart_uploads")
        return {"Uploads": self.uploads[:MaxUploads], "IsTruncated": False}

    def list_parts(
        self, Bucket: str, Key: str, UploadId: str, MaxParts: int, PartNumberMarker: int
    ) -> dict[str, Any]:
        self.calls.append("list_parts")
        return {"Parts": self.parts.get(UploadId, []), "IsTruncated": False}


class _Fixture:
    def __init__(self, s3: StubS3 | None = None) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.coverage = CoverageStore(self.rds, FAMILIES)
        self.s3 = s3 or StubS3()
        self.collector = AuditCollector(
            self.rds,
            self.coverage,
            StoreAdmin(self.s3, "bucket", _PREFIX),
            _LOGGER,
            AuditTrail(self.rds),
            min_age_sec=3600,
            page_pause_sec=0,
            clock=lambda: _NOW,
        )

    def certify_all(self, generation: int = 1) -> None:
        self.coverage.mark_hooks_on(generation)
        for family in FAMILIES:
            self.coverage.complete(generation, family)

    def condemned(self, scope: str = _A) -> dict[str, dict[str, Any]]:
        stored = cast(dict[str, str], self.raw.hgetall(reclaim_condemned_key(scope)))
        return {digest: json.loads(value) for digest, value in stored.items()}

    def audit(self, kind: str) -> list[dict[str, str]]:
        entries = cast(list[Any], self.raw.xrange(RECLAIM_AUDIT_KEY))
        return [fields for _, fields in entries if fields["kind"] == kind]


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def _assert_nothing_destroyed(s3: StubS3, raw: fakeredis.FakeRedis) -> None:
    assert not [c for c in s3.calls if c.startswith(("delete", "abort"))]
    assert not [k for k in raw.scan_iter("rc:*") if k.endswith(":deleting")]


def test_only_unbound_unheld_old_objects_are_condemned(fx) -> None:
    fx.certify_all()
    bound = fx.s3.put(_A, "bound")
    pinned = fx.s3.put(_A, "pinned")
    orphan = fx.s3.put(_A, "orphan", size=33)
    young = fx.s3.put(_A, "young", size=7, at=_NOW - 60)
    fx.raw.sadd(reclaim_protected_key(_A), bound)
    fx.raw.hset(reclaim_pins_key(_A), pinned, json.dumps([{"n": "1:1", "l": "{}"}]))
    fx.s3.versions.append(
        {
            "Key": f"{_PREFIX}/{_A}/staging/leftover",
            "VersionId": "v",
            "Size": 1,
            "LastModified": _when(_OLD),
            "IsLatest": True,
        }
    )

    result = fx.collector.sweep()

    assert set(fx.condemned()) == {orphan}
    entry = fx.condemned()[orphan]
    assert entry["version_id"] == "v-orphan" and entry["sweep"] == result.sweep
    (scope,) = result.scopes
    assert (scope.orphans, scope.orphan_bytes, scope.kept, scope.young) == (1, 33, 2, 1)
    assert scope.foreign == 1
    assert young not in fx.condemned()
    assert (result.young, result.young_bytes) == (1, 7)
    _assert_nothing_destroyed(fx.s3, fx.raw)


def test_a_retiring_object_is_never_condemned(fx) -> None:
    fx.certify_all()
    digest = fx.s3.put(_A, "retiring")
    fx.raw.hset(reclaim_deleting_key(_A), digest, "{}")

    fx.collector.sweep()

    assert fx.condemned() == {}


def test_a_tree_awaiting_expansion_stops_the_scope(fx) -> None:
    fx.certify_all()
    fx.s3.put(_A, "orphan")
    fx.raw.sadd(reclaim_unexpanded_key(_A), _digest("manifest"))

    result = fx.collector.sweep()

    assert fx.condemned() == {}
    assert not result.scopes[0].certified
    assert result.scopes[0].reason.startswith("unexpanded")


def test_an_uncertified_scope_is_listed_by_no_page(fx) -> None:
    fx.coverage.mark_hooks_on(1)
    fx.s3.put(_A, "orphan")
    fx.raw.sadd(RECLAIM_SCOPES_KEY, _A)

    result = fx.collector.sweep()

    assert fx.condemned() == {}
    assert result.scopes[0].reason.startswith("incomplete")
    assert fx.audit("scope")[0]["certified"] == "false"


def test_condemnation_rechecks_what_changed_since_the_page_was_read(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.certify_all()
    raced = fx.s3.put(_A, "raced")
    fenced = fx.s3.put(_A, "other")
    real_eval = fx.rds.sync.eval

    def bind_first(script: str, numkeys: int, *args: str) -> Any:
        if "condemned" in args[numkeys - 1]:
            fx.raw.sadd(reclaim_protected_key(_A), raced)
        return real_eval(script, numkeys, *args)

    monkeypatch.setattr(fx.rds.sync, "eval", bind_first)
    fx.collector.sweep()

    assert set(fx.condemned()) == {fenced}


def test_an_owed_protection_after_certification_stops_condemnation(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.certify_all()
    fx.s3.put(_A, "orphan")
    real_eval = fx.rds.sync.eval

    def owe_first(script: str, numkeys: int, *args: str) -> Any:
        if "condemned" in args[numkeys - 1]:
            fx.raw.set(reclaim_uncertified_key(_A), "1")
        return real_eval(script, numkeys, *args)

    monkeypatch.setattr(fx.rds.sync, "eval", owe_first)
    result = fx.collector.sweep()

    assert fx.condemned() == {}
    assert result.scopes[0].stopped == "uncertified"


def test_a_condemned_object_keeps_when_it_was_first_seen(fx) -> None:
    fx.certify_all()
    digest = fx.s3.put(_A, "orphan")
    fx.raw.hset(
        reclaim_condemned_key(_A),
        digest,
        json.dumps({"version_id": "v-orphan", "sweep": "0:1", "seen_at": "5.000"}),
    )

    fx.collector.sweep()

    assert fx.condemned()[digest]["seen_at"] == "5.000"


def test_every_page_is_swept_one_at_a_time() -> None:
    fx = _Fixture(StubS3(page=3))
    fx.certify_all()
    digests = {fx.s3.put(_A, f"orphan-{i}") for i in range(10)}

    result = fx.collector.sweep()

    assert set(fx.condemned()) == digests
    assert result.scopes[0].versions == 10
    assert fx.s3.calls.count("list_object_versions") >= 4


def test_open_uploads_are_reported_and_left_open(fx) -> None:
    fx.certify_all()
    fx.s3.uploads = [
        {
            "Key": f"{_PREFIX}/{_A}/sha256/{_digest('big')}",
            "UploadId": "u-1",
            "Initiated": _when(_OLD),
        }
    ]
    fx.s3.parts = {"u-1": [{"Size": 5 << 20}, {"Size": 1 << 20}]}

    result = fx.collector.sweep()

    assert result.uploads is not None
    assert (result.uploads.uploads, result.uploads.parts, result.uploads.bytes) == (
        1,
        2,
        6 << 20,
    )
    assert result.uploads.by_scope == {_A: 1}
    _assert_nothing_destroyed(fx.s3, fx.raw)


@pytest.mark.parametrize("status", [None, "Suspended"])
def test_an_unversioned_bucket_is_ineligible_and_keeps_everything(status) -> None:
    fx = _Fixture(StubS3(versioning=status))
    fx.certify_all()
    fx.s3.put(_A, "orphan")

    result = fx.collector.sweep()

    assert result.eligibility.reason is Ineligibility.UNVERSIONED
    assert fx.condemned() == {}
    assert "list_object_versions" not in fx.s3.calls
    assert fx.audit("ineligible")[0]["reason"] == Ineligibility.UNVERSIONED.value


def test_a_filesystem_store_is_ineligible_and_keeps_everything() -> None:
    server = fakeredis.FakeServer()
    rds = fake_redis_client(server)
    coverage = CoverageStore(rds, FAMILIES)
    store = ObjectStoreConfig(backend=BACKEND_FILESYSTEM)
    assert StoreAdmin.from_config(store) is None
    collector = AuditCollector(
        rds, coverage, StoreAdmin.from_config(store), _LOGGER, AuditTrail(rds)
    )

    result = collector.sweep()

    assert result.eligibility.reason is Ineligibility.FILESYSTEM
    assert result.scopes == []


def test_a_credential_that_cannot_list_is_ineligible(fx, monkeypatch) -> None:
    def refuse(**_: Any) -> Any:
        raise PermissionError("AccessDenied")

    monkeypatch.setattr(fx.s3, "list_multipart_uploads", refuse, raising=False)
    fx.certify_all()

    result = fx.collector.sweep()

    assert result.eligibility.reason is Ineligibility.CANNOT_LIST

"""Multipart record upkeep: a record leaves only once its owner has ended and the store,
asked about its exact UploadId, no longer holds the upload."""

import logging
import threading
from typing import Any, cast

import fakeredis
import pytest
from botocore.exceptions import ClientError

from server.clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    content_finalization_key,
    content_finalization_scope_key,
    reclaim_multipart_key,
)
from server.reclamation.multipart import MultipartOwner, MultipartStore
from server.reclamation.store_admin import StoreAdmin
from server.reclamation.upkeep import MultipartUpkeep
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_PREFIX = "fabric"
_LOGGER = logging.getLogger("test.reclamation.upkeep")


def _key(upload_id: str) -> str:
    return f"{_PREFIX}/{_A}/sha256/{upload_id}"


class SiloS3:
    """A MinIO-family store: listing open uploads by prefix names none, listing them
    by bucket reads a cache a restart empties, and only ``ListParts`` by UploadId
    answers exactly."""

    def __init__(self) -> None:
        self.open: dict[str, str] = {}
        self.cached: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.unreachable = False

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the upkeep called {name}")

    def create(self, upload_id: str) -> None:
        self.open[upload_id] = _key(upload_id)
        self.cached.add(upload_id)

    def close(self, upload_id: str) -> None:
        del self.open[upload_id]
        self.cached.discard(upload_id)

    def restart(self) -> None:
        self.cached.clear()

    def list_multipart_uploads(
        self, Bucket: str, Prefix: str, MaxUploads: int, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(("list_multipart_uploads", Prefix))
        listed = [] if Prefix else sorted(self.cached)
        return {
            "Uploads": [{"Key": self.open[u], "UploadId": u} for u in listed],
            "IsTruncated": False,
        }

    def list_parts(
        self, Bucket: str, Key: str, UploadId: str, MaxParts: int, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(("list_parts", UploadId))
        if self.unreachable:
            raise ConnectionError("store unreachable")
        if self.open.get(UploadId) != Key:
            raise ClientError(
                {"Error": {"Code": "NoSuchUpload", "Message": "gone"}}, "ListParts"
            )
        return {"Parts": [], "IsTruncated": False}

    def checked(self) -> list[str]:
        return [upload_id for call, upload_id in self.calls if call == "list_parts"]


class _Fixture:
    def __init__(self, max_checks: int = 500) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.s3 = SiloS3()
        self.store = MultipartStore(self.rds)
        self.live: set[tuple[str, str, str]] = set()
        self.upkeep = MultipartUpkeep(
            self.rds,
            self.store,
            StoreAdmin(self.s3, "bucket", _PREFIX),
            self.owner_of,
            _LOGGER,
            max_checks=max_checks,
            page_pause_sec=0,
        )

    def owner_of(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> tuple[str, MultipartOwner] | None:
        if (task_id, worker_id, dispatch_id or "") in self.live:
            return _A, MultipartOwner.dispatch(dispatch_id or "", task_id)
        return None

    def register(self, upload_id: str, *, live: bool, opened: bool = True) -> None:
        if opened:
            self.s3.create(upload_id)
        task_id, dispatch_id = f"tsk-{upload_id}", f"dsp-{upload_id}"
        owner = MultipartOwner.dispatch(dispatch_id, task_id)
        self.store.register(_A, upload_id, _key(upload_id), owner, "wkr-1")
        if live:
            self.live.add((task_id, "wkr-1", dispatch_id))

    def end(self, upload_id: str) -> None:
        self.live.discard((f"tsk-{upload_id}", "wkr-1", f"dsp-{upload_id}"))

    def records(self) -> set[str]:
        return set(cast(list[str], self.raw.hkeys(reclaim_multipart_key(_A))))


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def test_records_of_open_uploads_survive_a_store_that_lists_none(fx) -> None:
    fx.register("u-open", live=False)
    fx.register("u-closed", live=False)
    fx.register("u-live", live=True)
    fx.s3.close("u-closed")
    fx.s3.close("u-live")
    fx.s3.restart()

    result = fx.upkeep.run_pass()

    assert fx.records() == {"u-open", "u-live"}
    assert sorted(fx.s3.checked()) == ["u-closed", "u-open"]
    assert (result.ended, result.dropped, result.open) == (2, 1, 1)
    assert result.complete


def test_a_live_owners_record_is_never_checked(fx) -> None:
    fx.register("u-live", live=True)

    for _ in range(3):
        fx.upkeep.run_pass()

    assert fx.s3.checked() == []
    assert fx.records() == {"u-live"}


def test_an_ended_owners_open_upload_is_checked_once(fx) -> None:
    fx.register("u-1", live=False)

    fx.upkeep.run_pass()
    second = fx.upkeep.run_pass()

    assert fx.s3.checked() == ["u-1"]
    assert second.open == 1 and second.checked == 0
    assert fx.records() == {"u-1"}


def test_an_owner_that_ends_again_is_checked_again(fx) -> None:
    fx.register("u-1", live=False)
    fx.upkeep.run_pass()
    fx.live.add(("tsk-u-1", "wkr-1", "dsp-u-1"))
    fx.upkeep.run_pass()
    fx.end("u-1")
    fx.s3.close("u-1")

    fx.upkeep.run_pass()

    assert fx.s3.checked() == ["u-1", "u-1"]
    assert fx.records() == set()


def test_a_pass_checks_at_most_its_budget() -> None:
    fx = _Fixture(max_checks=2)
    for i in range(5):
        fx.register(f"u-{i}", live=False, opened=False)

    first = fx.upkeep.run_pass()

    assert (first.checked, first.dropped, first.deferred) == (2, 2, 3)
    assert not first.complete
    fx.upkeep.run_pass()
    fx.upkeep.run_pass()
    assert fx.records() == set()


def test_a_store_without_an_answer_drops_nothing_and_stops_asking(fx) -> None:
    for upload_id in ("u-1", "u-2"):
        fx.register(upload_id, live=False, opened=False)
    fx.s3.unreachable = True

    result = fx.upkeep.run_pass()

    assert fx.records() == {"u-1", "u-2"}
    assert len(fx.s3.checked()) == 1
    assert result.deferred == 2 and not result.complete


@pytest.mark.parametrize(
    ("assigned", "recorded", "checked"),
    [(_A, False, []), (_A, True, ["u-1"]), (None, False, ["u-1"])],
    ids=["unrecorded", "recorded", "unassigned"],
)
def test_a_finalization_owns_its_upload_until_its_winner_is_recorded(
    fx, assigned: str | None, recorded: bool, checked: list[str]
) -> None:
    fx.raw.set(content_finalization_scope_key("idm-1"), _A)
    fx.s3.create("u-1")
    fx.store.register(
        _A, "u-1", _key("u-1"), MultipartOwner.finalization("idm-1"), "wkr-1"
    )
    if assigned is None:
        fx.raw.delete(content_finalization_scope_key("idm-1"))
    if recorded:
        fx.raw.set(content_finalization_key(_A, "idm-1"), "{}")

    fx.upkeep.run_pass()

    assert fx.s3.checked() == checked
    assert fx.records() == {"u-1"}


def test_the_upkeep_stops_once_a_later_root_starts(fx) -> None:
    fx.register("u-1", live=False, opened=False)
    fx.raw.set(RECLAIM_ROOT_EPOCH_KEY, "8")

    fx.upkeep.run(lambda: 7, threading.Event(), interval_sec=0)

    assert fx.s3.checked() == []
    assert fx.records() == {"u-1"}


def test_the_upkeep_runs_under_its_own_root(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.register("u-1", live=False, opened=False)
    fx.raw.set(RECLAIM_ROOT_EPOCH_KEY, "7")
    stop = threading.Event()
    real_pass = fx.upkeep.run_pass

    def pass_once(stop_: threading.Event | None = None) -> Any:
        stop.set()
        return real_pass(stop_)

    monkeypatch.setattr(fx.upkeep, "run_pass", pass_once)
    fx.upkeep.run(lambda: 7, stop, interval_sec=0)

    assert fx.records() == set()

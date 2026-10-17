"""Multipart registration: the verdict on each UploadId and who owns it."""

import json
import logging
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    content_finalization_key,
    content_finalization_scope_key,
    reclaim_multipart_dispatch_key,
    reclaim_multipart_key,
    worker_key,
)
from server.reclamation.multipart import (
    MultipartOwner,
    MultipartRegistrations,
    MultipartStore,
    OwnerKind,
)
from shared.content.multipart import MULTIPART_ACK_FRAME_KIND, MultipartVerdict
from shared.schemas.command import MediatedOpMessage
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"
_KEY = "org-a/sha256/abc"
_DISPATCH = MultipartOwner.dispatch("dsp-1", "tsk-1")


@pytest.fixture
def store() -> tuple[MultipartStore, fakeredis.FakeRedis]:
    server = fakeredis.FakeServer()
    return (
        MultipartStore(fake_redis_client(server)),
        fakeredis.FakeRedis(server=server, decode_responses=True),
    )


def _record(raw: fakeredis.FakeRedis, upload_id: str) -> dict[str, Any]:
    value = cast(str | None, raw.hget(reclaim_multipart_key(_SCOPE), upload_id))
    assert value is not None
    return json.loads(value)


def test_a_new_upload_is_registered_under_its_owner(store) -> None:
    multipart, raw = store

    result = multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1", now=12.5)

    assert result.verdict is MultipartVerdict.ACK
    assert _record(raw, "up-1") == {
        "key": _KEY,
        "state": "registered",
        "owner": {"kind": "dispatch", "dispatch_id": "dsp-1", "task_id": "tsk-1"},
        "registration": "wkr-1",
        "initiated_at": 12500,
    }
    assert raw.hkeys(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1")) == ["up-1"]


def test_a_repeated_registration_is_acknowledged_with_the_first_record(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1", now=1.0)

    again = multipart.register(
        _SCOPE, "up-1", _KEY, MultipartOwner.upload("dsp-1"), "wkr-1", now=9.0
    )

    assert again.verdict is MultipartVerdict.ACK
    assert again.record == _record(raw, "up-1")
    assert _record(raw, "up-1")["initiated_at"] == 1000
    assert _record(raw, "up-1")["owner"]["kind"] == "dispatch"


@pytest.mark.parametrize("state", ["aborting", "aborted"])
def test_an_upload_being_aborted_is_doomed(store, state: str) -> None:
    multipart, raw = store
    raw.hset(
        reclaim_multipart_key(_SCOPE),
        "up-1",
        json.dumps({"key": _KEY, "state": state, "registration": "wkr-1"}),
    )

    result = multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")

    assert result.verdict is MultipartVerdict.DOOMED
    assert _record(raw, "up-1")["state"] == state


def test_an_abort_intent_without_a_record_still_dooms_the_upload(store) -> None:
    multipart, raw = store
    raw.hset(
        reclaim_multipart_key(_SCOPE),
        "up-9",
        json.dumps({"key": _KEY, "state": "aborting"}),
    )

    assert (
        multipart.register(_SCOPE, "up-9", _KEY, _DISPATCH, "wkr-2").verdict
        is MultipartVerdict.DOOMED
    )


def test_another_registration_naming_a_registered_upload_owns_nothing(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")

    other = multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-2")
    other_key = multipart.register(
        _SCOPE, "up-1", "org-a/sha256/def", _DISPATCH, "wkr-1"
    )

    assert other.verdict is MultipartVerdict.UNOWNED
    assert other_key.verdict is MultipartVerdict.UNOWNED
    assert _record(raw, "up-1")["registration"] == "wkr-1"


def test_a_finalization_owns_an_upload_only_while_its_winner_is_unrecorded(
    store,
) -> None:
    multipart, raw = store
    owner = MultipartOwner.finalization("idm-1")

    unassigned = multipart.register(_SCOPE, "up-1", _KEY, owner, "wkr-1")
    raw.set(content_finalization_scope_key("idm-1"), "org-b")
    other_scope = multipart.register(_SCOPE, "up-1", _KEY, owner, "wkr-1")
    raw.set(content_finalization_scope_key("idm-1"), _SCOPE)
    live = multipart.register(_SCOPE, "up-1", _KEY, owner, "wkr-1")
    raw.set(content_finalization_key(_SCOPE, "idm-1"), "{}")
    won = multipart.register(_SCOPE, "up-2", _KEY, owner, "wkr-1")

    assert unassigned.verdict is MultipartVerdict.UNOWNED
    assert other_scope.verdict is MultipartVerdict.UNOWNED
    assert live.verdict is MultipartVerdict.ACK
    assert _record(raw, "up-1")["owner"] == {"kind": "finalization", "idm": "idm-1"}
    assert won.verdict is MultipartVerdict.UNOWNED
    assert not raw.hexists(reclaim_multipart_key(_SCOPE), "up-2")


def test_a_transfer_moves_only_the_dispatchs_live_uploads(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1", now=1.0)
    multipart.register(_SCOPE, "up-2", _KEY, _DISPATCH, "wkr-1", now=2.0)
    multipart.register(
        _SCOPE, "up-3", _KEY, MultipartOwner.dispatch("dsp-2", "tsk-2"), "wkr-1"
    )
    aborting = json.loads(cast(str, raw.hget(reclaim_multipart_key(_SCOPE), "up-2")))
    aborting["state"] = "aborting"
    raw.hset(reclaim_multipart_key(_SCOPE), "up-2", json.dumps(aborting))
    upload = MultipartOwner.upload("dsp-1")

    moved = multipart.transfer(_SCOPE, "dsp-1", upload)

    assert moved == 1
    assert _record(raw, "up-1") == {
        "key": _KEY,
        "state": "registered",
        "owner": {"kind": "upload", "dispatch_id": "dsp-1"},
        "registration": "wkr-1",
        "initiated_at": 1000,
    }
    assert _record(raw, "up-2")["owner"]["kind"] == "dispatch"
    assert _record(raw, "up-3")["owner"]["dispatch_id"] == "dsp-2"
    assert multipart.transfer(_SCOPE, "dsp-1", upload) == 0


def test_a_registration_racing_its_dispatchs_transfer_takes_the_new_owner(
    store,
) -> None:
    multipart, raw = store
    multipart.transfer(_SCOPE, "dsp-1", MultipartOwner.upload("dsp-1"))

    late = multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")

    assert late.verdict is MultipartVerdict.ACK
    assert _record(raw, "up-1")["owner"] == {"kind": "upload", "dispatch_id": "dsp-1"}
    assert raw.hkeys(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1")) == ["~moved"]


def test_a_corrupt_record_refuses_a_transfer_without_moving_any(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")
    raw.hset(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"), "up-2", "1")
    raw.hset(reclaim_multipart_key(_SCOPE), "up-2", "not json")

    with pytest.raises(Exception):
        multipart.transfer(_SCOPE, "dsp-1", MultipartOwner.upload("dsp-1"))

    assert _record(raw, "up-1")["owner"]["kind"] == "dispatch"
    assert not raw.hexists(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"), "~moved")


def test_a_key_of_the_wrong_type_refuses_before_writing(store) -> None:
    multipart, raw = store
    raw.set(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"), "x")

    with pytest.raises(Exception, match="WRONGTYPE"):
        multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")

    assert not raw.exists(reclaim_multipart_key(_SCOPE))


def test_a_prune_drops_only_closed_uploads_registered_before_the_listing(
    store,
) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-open", _KEY, _DISPATCH, "wkr-1", now=1.0)
    multipart.register(_SCOPE, "up-closed", _KEY, _DISPATCH, "wkr-1", now=2.0)
    multipart.register(_SCOPE, "up-late", _KEY, _DISPATCH, "wkr-1", now=9.0)
    raw.hset(reclaim_multipart_key(_SCOPE), "up-corrupt", "not json")

    assert multipart.prune(_SCOPE, {"up-open"}, registered_before=5.0) == 1

    assert set(raw.hkeys(reclaim_multipart_key(_SCOPE))) == {
        "up-open",
        "up-late",
        "up-corrupt",
    }
    assert set(raw.hkeys(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"))) == {
        "up-open",
        "up-late",
    }


def test_the_last_prune_of_a_dispatchs_uploads_leaves_no_index(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1", now=1.0)

    multipart.prune(_SCOPE, set(), registered_before=5.0)

    assert not raw.exists(reclaim_multipart_key(_SCOPE))
    assert not raw.exists(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"))


def test_a_released_dispatch_leaves_no_index_and_keeps_its_records(store) -> None:
    multipart, raw = store
    multipart.register(_SCOPE, "up-1", _KEY, _DISPATCH, "wkr-1")
    multipart.transfer(_SCOPE, "dsp-1", MultipartOwner.upload("dsp-1"))

    multipart.release(_SCOPE, "dsp-1")

    assert not raw.exists(reclaim_multipart_dispatch_key(_SCOPE, "dsp-1"))
    assert _record(raw, "up-1")["owner"] == {"kind": "upload", "dispatch_id": "dsp-1"}


def test_the_content_protocol_a_worker_registered_with_is_read_back(store) -> None:
    multipart, raw = store
    raw.hset(worker_key("wkr-1"), mapping={"id": "wkr-1", "content_protocol": "1"})
    raw.hset(worker_key("wkr-0"), mapping={"id": "wkr-0"})

    assert multipart.content_protocol("wkr-1") == 1
    assert multipart.content_protocol("wkr-0") == 0
    assert multipart.content_protocol("wkr-missing") == 0


class _Workers:
    def __init__(self) -> None:
        self.sent: list[MediatedOpMessage] = []

    def get_worker(self, worker_id: str) -> Any:
        return None if worker_id == "wkr-gone" else object()

    def publish_mediated_op(self, worker: Any, message: MediatedOpMessage) -> int:
        self.sent.append(message)
        return 1


def _registrations(
    multipart: MultipartStore,
    owner: tuple[str, MultipartOwner] | None,
) -> tuple[MultipartRegistrations, _Workers, list[tuple[str, str, str | None]]]:
    asked: list[tuple[str, str, str | None]] = []

    def owner_of(
        task_id: str, worker_id: str, dispatch_id: str | None
    ) -> tuple[str, MultipartOwner] | None:
        asked.append((task_id, worker_id, dispatch_id))
        return owner

    workers = _Workers()
    return (
        MultipartRegistrations(
            multipart, owner_of, cast(Any, workers), logging.getLogger("test")
        ),
        workers,
        asked,
    )


def _payload(**extra: Any) -> dict[str, Any]:
    return {
        "task_id": "tsk-1",
        "dispatch_id": "dsp-1",
        "scope": _SCOPE,
        "key": _KEY,
        "upload_id": "up-1",
        **extra,
    }


def _answers(workers: _Workers) -> list[tuple[str, dict[str, Any]]]:
    return [(m.frame_kind, m.payload) for m in workers.sent]


def test_control_acknowledges_an_upload_its_runtime_owner_holds(store) -> None:
    multipart, raw = store
    service, workers, asked = _registrations(multipart, (_SCOPE, _DISPATCH))

    service.on_registration("wkr-1", _payload())

    assert asked == [("tsk-1", "wkr-1", "dsp-1")]
    assert _answers(workers) == [
        (MULTIPART_ACK_FRAME_KIND, {"upload_id": "up-1", "verdict": "ack"})
    ]
    assert _record(raw, "up-1")["owner"]["kind"] == OwnerKind.DISPATCH.value


@pytest.mark.parametrize(
    "owner",
    [None, ("org-b", _DISPATCH)],
    ids=["no-owner", "other-scope"],
)
def test_control_refuses_an_upload_no_live_owner_holds_and_records_nothing(
    store, owner: tuple[str, MultipartOwner] | None
) -> None:
    multipart, raw = store
    service, workers, _ = _registrations(multipart, owner)

    service.on_registration("wkr-1", _payload())

    assert _answers(workers) == [
        (MULTIPART_ACK_FRAME_KIND, {"upload_id": "up-1", "verdict": "unowned"})
    ]
    assert not raw.exists(reclaim_multipart_key(_SCOPE))


def test_a_named_finalization_is_checked_by_control_not_the_runtime(store) -> None:
    multipart, raw = store
    raw.set(content_finalization_scope_key("idm-1"), _SCOPE)
    service, workers, asked = _registrations(multipart, None)

    service.on_registration("wkr-1", _payload(idempotency_key="idm-1"))

    assert asked == []
    assert _answers(workers)[0][1]["verdict"] == "ack"
    assert _record(raw, "up-1")["owner"] == {"kind": "finalization", "idm": "idm-1"}


def test_a_store_that_cannot_decide_leaves_the_registration_unanswered(
    store,
) -> None:
    multipart, raw = store
    raw.set(reclaim_multipart_key(_SCOPE), "not a hash")
    service, workers, _ = _registrations(multipart, (_SCOPE, _DISPATCH))

    service.on_registration("wkr-1", _payload())

    assert workers.sent == []

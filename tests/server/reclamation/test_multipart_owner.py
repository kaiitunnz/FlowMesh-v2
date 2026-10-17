"""The runtime names a multipart upload's owner and moves it to the owed upload."""

import json
import logging
from typing import Any, cast

import fakeredis

from server.clients.redis import reclaim_multipart_dispatch_key, reclaim_multipart_key
from server.reclamation.multipart import MultipartOwner, MultipartStore
from server.reclamation.store_admin import StoreAdmin
from server.reclamation.upkeep import MultipartUpkeep
from shared.tasks.artifact_upload import UploadReport
from tests.server.reclamation.test_upkeep import SiloS3
from tests.server.redis_helpers import fake_redis_client
from tests.server.task.test_artifact_uploads import _dispatched, _Setup


def _wired() -> tuple[_Setup, MultipartStore, fakeredis.FakeRedis]:
    setup = _dispatched()
    store = MultipartStore(fake_redis_client(setup.server))
    return setup, store, fakeredis.FakeRedis(server=setup.server, decode_responses=True)


def _owner(raw: fakeredis.FakeRedis, upload_id: str) -> dict[str, Any]:
    value = cast(str, raw.hget(reclaim_multipart_key("org"), upload_id))
    return json.loads(value)


def test_the_dispatch_holding_a_running_task_owns_its_uploads() -> None:
    setup, _, _ = _wired()
    owner = setup.runtime.multipart_owner

    assert owner(setup.train, "wkr-1", "dsp-1") == (
        "org",
        MultipartOwner.dispatch("dsp-1", setup.train),
    )
    assert owner(setup.train, "wkr-2", "dsp-1") is None
    assert owner(setup.train, "wkr-1", "dsp-0") is None
    assert owner(setup.train, "wkr-1", None) is None


def test_a_staged_success_moves_its_dispatchs_uploads_to_the_owed_upload() -> None:
    setup, store, raw = _wired()
    scope, owner = cast(
        tuple[str, MultipartOwner],
        setup.runtime.multipart_owner(setup.train, "wkr-1", "dsp-1"),
    )
    store.register(scope, "up-1", "org/sha256/a", owner, "wkr-1", now=3.0)

    setup.succeed([setup.tree(stored=False)])

    assert _owner(raw, "up-1")["owner"] == {
        "kind": "upload",
        "dispatch_id": "dsp-1",
        "task_id": setup.train,
    }
    assert _owner(raw, "up-1")["initiated_at"] == 3000
    assert _owner(raw, "up-1")["registration"] == "wkr-1"
    assert setup.runtime.multipart_owner(setup.train, "wkr-1", "dsp-1") == (
        "org",
        MultipartOwner.upload("dsp-1"),
    )
    assert setup.runtime.multipart_owner(setup.train, "wkr-2", "dsp-1") is None


def test_a_settled_upload_owns_nothing_more() -> None:
    setup, store, raw = _wired()
    scope, owner = cast(
        tuple[str, MultipartOwner],
        setup.runtime.multipart_owner(setup.train, "wkr-1", "dsp-1"),
    )
    store.register(scope, "up-1", "org/sha256/a", owner, "wkr-1")
    setup.succeed([setup.file(b"weights", "weights.bin", stored=True)])
    assert raw.exists(reclaim_multipart_dispatch_key("org", "dsp-1"))

    setup.report(UploadReport.UPLOADED)

    assert setup.runtime.multipart_owner(setup.train, "wkr-1", "dsp-1") is None
    assert not raw.exists(reclaim_multipart_dispatch_key("org", "dsp-1"))
    # Its record stays until the store no longer holds the upload open.
    assert _owner(raw, "up-1")["owner"]["kind"] == "upload"


def test_a_success_owing_no_upload_moves_nothing() -> None:
    setup, store, raw = _wired()
    store.register(
        "org",
        "up-1",
        "org/sha256/a",
        MultipartOwner.dispatch("dsp-1", setup.train),
        "wkr-1",
    )

    setup.succeed([])

    assert _owner(raw, "up-1")["owner"]["kind"] == "dispatch"


def test_the_upkeep_checks_an_upload_only_once_its_runtime_owner_ends() -> None:
    setup, store, raw = _wired()
    s3 = SiloS3()
    upkeep = MultipartUpkeep(
        fake_redis_client(setup.server),
        store,
        StoreAdmin(s3, "bucket"),
        setup.runtime.multipart_owner,
        logging.getLogger("test"),
        page_pause_sec=0,
    )
    scope, owner = cast(
        tuple[str, MultipartOwner],
        setup.runtime.multipart_owner(setup.train, "wkr-1", "dsp-1"),
    )
    store.register(scope, "up-1", "org/sha256/a", owner, "wkr-1")

    upkeep.run_pass()
    setup.succeed([setup.file(b"weights", "weights.bin", stored=True)])
    upkeep.run_pass()
    assert s3.checked() == []

    setup.report(UploadReport.UPLOADED)
    upkeep.run_pass()

    assert s3.checked() == ["up-1"]
    assert not raw.exists(reclaim_multipart_key("org"))

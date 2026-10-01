"""A worker moving to a new registration leaves behind everything the previous one
held: the dispatch it ran, the requests its boundaries captured, its store access, and
its replicas' claim gates."""

import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.content import (
    BACKEND_FILESYSTEM,
    ContentOperationKind,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ObjectStoreConfig,
    ScopedContentCredential,
)
from shared.resident.contracts import ReplicaEndpoint
from shared.tools.search.schema import SEARCH_INTERFACE, ToolRequest
from tests.worker.factories import make_worker_config, make_worker_hardware
from worker.content import ContentAccessRegistry, WorkerContentPlane
from worker.content.access import ContentAccessDenied
from worker.executors.base_executor import EchoExecutor
from worker.lifecycle import Lifecycle
from worker.resident.replica_sidecar import ResidentReplicaSidecar
from worker.runner import Runner


def _access(task_id: str) -> ContentStoreAccess:
    return ContentStoreAccess(
        grant=ContentStoreAccessGrant(
            grant_id="csg-1",
            task_id=task_id,
            authorization_scope="tenant-a",
            subject="wkr-1",
            subject_generation=1,
            operations=(ContentOperationKind.READ, ContentOperationKind.WRITE),
            backend_policy_version="test-1",
            expires_at_epoch=time.time() + 900,
        ),
        credential=ScopedContentCredential(material={"token": "opaque"}),
    )


def test_a_re_registered_worker_drops_what_its_previous_registration_held(
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.worker_id = "wkr-2"
    client.incarnation = 2
    lifecycle = Lifecycle(client, 30, 120, tmp_path / "worker.hb", 1.0)
    abandoned: list[str | None] = []
    lifecycle.set_abandon_handler(abandoned.append)
    lifecycle.pending_egress_requests.put(
        "tsk-1",
        "c0",
        ToolRequest(interface=SEARCH_INTERFACE, query="q", max_results=1),
    )
    lifecycle.resident_requests.put("tsk-1", "c1", "request")
    access = ContentAccessRegistry(
        ObjectStoreConfig(backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path),
        arrival_wait_sec=0.05,
    )
    access.accept(_access("tsk-1"))
    lifecycle.content_plane = WorkerContentPlane(None, access)

    lifecycle._on_reregistered("dsp-1")

    assert abandoned == ["dsp-1"]
    assert lifecycle.pending_egress_requests.occurrences() == []
    assert lifecycle.resident_requests.occurrences() == []
    with pytest.raises(ContentAccessDenied):
        access.store_for("tsk-1", "tenant-a")


def test_abandoning_a_dispatch_unbinds_the_replicas_served_before(
    tmp_path: Path,
) -> None:
    executor = EchoExecutor(make_worker_config())
    runner = Runner(
        lifecycle=cast(Any, MagicMock()),
        task_stream=[],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"echo": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    resident_host = MagicMock()
    runner._resident_host = resident_host

    runner.abandon_running(None)

    resident_host.unbind_replicas.assert_called_once_with()


def test_unbinding_every_replica_closes_each_claim_gate() -> None:
    sidecar = ResidentReplicaSidecar(sink=MagicMock(), engine_open=MagicMock())
    for replica_id in ("rpl-1", "rpl-2"):
        sidecar.bind(
            replica_id=replica_id,
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )

    sidecar.unbind_all()

    assert sidecar._bindings == {}

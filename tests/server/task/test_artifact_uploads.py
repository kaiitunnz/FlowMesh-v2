"""A success that staged its artifacts leaves their upload owed: the task is DONE at
once, its workflow stays open until the upload settles, and an owner that leaves has
the upload settled from the store."""

import asyncio
import hashlib
import logging
import time
from types import SimpleNamespace
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    task_state_key,
    workflow_sources_key,
    workflow_uploads_key,
)
from server.config import OrchestrationConfig
from server.registries.task_states import load_task_state
from server.registries.workflow import WorkflowRegistry, WorkflowStatus
from server.task.models import TaskRecord, TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from server.task.runtime.artifact_uploads import UploadReconciler, manual_reconciler
from shared.content import (
    TREE_MEDIA_TYPE,
    ContentReference,
    ContentUnavailable,
    TreeManifest,
    TreeMember,
)
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.artifact_upload import (
    ARTIFACTS_PENDING,
    ArtifactDurability,
    UploadDisposition,
    UploadReport,
    UploadSettled,
)
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.result_store import (
    StoreResultReader,
    make_result_reader,
    result_payload,
    store_result,
)
from tests.server.task.test_v2_orchestration import _TS

_V1 = """apiVersion: flowmesh/v1
kind: EchoTask
metadata: {name: uploads}
spec:
  taskType: echo
  stages:
    - name: train
      spec:
        data: {type: list, items: [a]}
    - name: use
      dependsOn: [train]
      spec:
        data: {type: list, items: [b]}
"""

_V2 = """apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: uploads}
spec:
  graph:
    nodes:
      - name: train
        spec: {taskType: echo, data: {type: list, items: [a]}}
      - name: use
        dependsOn: [train]
        spec: {taskType: echo, data: {type: list, items: [b]}}
"""


class _Workers:
    """The worker registry: who is registered, and the answers sent to each."""

    def __init__(self) -> None:
        self.registered = {"wkr-1", "wkr-2"}
        self.sent: list[tuple[str, UploadSettled]] = []

    def get_worker(self, worker_id: str) -> Any:
        if worker_id not in self.registered:
            return None
        return SimpleNamespace(id=worker_id, node_id="nde-1", incarnation=1)

    def publish_mediated_op(self, worker: Any, message: Any) -> int:
        self.sent.append((worker.id, UploadSettled.model_validate(message.payload)))
        return 1

    async def get_workers_heard_async(self, ids: list[str]) -> list[Any]:
        return [(self.get_worker(i), False) for i in ids]

    def publish_interrupt(self, *args: Any) -> int:
        return 0

    def publish_revoke(self, *args: Any) -> int:
        return 0

    def release_worker(self, *args: Any) -> bool:
        return False

    def reservations(self) -> list[Any]:
        return []

    def answers(self, task_id: str) -> list[UploadDisposition]:
        return [s.disposition for _, s in self.sent if s.task_id == task_id]


def _runtime(
    registry: WorkflowRegistry,
    workers: _Workers,
    reader: StoreResultReader,
    reconciler: Any = manual_reconciler,
) -> TaskRuntime:
    return TaskRuntime(
        registry,
        cast(Any, workers),
        OrchestrationConfig(),
        reader,
        logging.getLogger("test.artifact_uploads"),
        credential_vault=InMemoryCredentialVault(),
        redrive=lambda fire, logger: StoreRedriveScheduler(
            fire, logger, run_thread=False
        ),
        reconciler=reconciler,
    )


class _Setup:
    def __init__(self, spec: str = _V1, reconciler: Any = manual_reconciler) -> None:
        self.server = fakeredis.FakeServer()
        self.registry = WorkflowRegistry(fake_redis_client(self.server))
        self.workers = _Workers()
        self.reader = make_result_reader()
        self.runtime = _runtime(self.registry, self.workers, self.reader, reconciler)
        workflow_id, results = asyncio.run(
            self.runtime.register("owner", "org", spec, format="native")
        )
        self.workflow_id = workflow_id
        self.train, self.use = (r.task_id for r in results)

    def tree(self, stored: bool) -> ArtifactBinding:
        """A tree of two members whose manifest is stored, and its members too when
        ``stored``."""
        store = self.reader.store
        members = [b"weights", b"config"]
        if stored:
            for data in members:
                store.write("org", data)
        manifest = TreeManifest.of(
            [
                TreeMember(f"f{i}", hashlib.sha256(data).hexdigest(), len(data))
                for i, data in enumerate(members)
            ]
        )
        reference = store.write("org", manifest.encode(), media_type=TREE_MEDIA_TYPE)
        return ArtifactBinding(
            name="final_model",
            kind=ArtifactKind.TREE,
            reference=reference,
            size_bytes=manifest.size_bytes,
            files=2,
        )

    def file(self, data: bytes, name: str, stored: bool) -> ArtifactBinding:
        reference = (
            self.reader.store.write("org", data)
            if stored
            else ContentReference(
                authorization_scope="org",
                content_digest=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
        return ArtifactBinding(
            name=name,
            kind=ArtifactKind.FILE,
            reference=reference,
            size_bytes=len(data),
        )

    def succeed(
        self, artifacts: list[ArtifactBinding], dispatch_id: str = "dsp-1"
    ) -> None:
        payload = result_payload(self.reader, self.train, {"ok": True}, "org")
        payload["artifacts"] = [a.model_dump(mode="json") for a in artifacts]
        payload[ARTIFACTS_PENDING] = True
        self.runtime.mark_succeeded(
            self.train, "wkr-1", payload, _TS, dispatch_id=dispatch_id
        )

    def report(
        self,
        report: UploadReport,
        worker_id: str = "wkr-1",
        dispatch_id: str = "dsp-1",
    ) -> None:
        self.runtime.record_artifact_upload(self.train, worker_id, dispatch_id, report)

    def status(self) -> tuple[WorkflowStatus, int, str | None]:
        workflow = self.registry.get_workflow(self.workflow_id)
        assert workflow is not None
        return (
            workflow.status,
            workflow.pending_artifact_uploads,
            workflow.failure,
        )

    def record(self, task_id: str) -> TaskRecord:
        record = self.runtime.get_record(task_id)
        assert record is not None
        return record

    def settled(self) -> bool:
        return self.runtime.workflow_settlement(self.workflow_id).settled

    def finish_use(self) -> None:
        record_dispatch(self.runtime, self.use, "wkr-2", dispatch_id="dsp-2")
        payload = result_payload(self.reader, self.use, {"ok": True}, "org")
        self.runtime.mark_succeeded(self.use, "wkr-2", payload, _TS, "dsp-2")


def _dispatched(spec: str = _V1) -> _Setup:
    setup = _Setup(spec)
    record_dispatch(setup.runtime, setup.train, dispatch_id="dsp-1")
    return setup


@pytest.mark.parametrize("spec", [_V1, _V2], ids=["v1", "v2"])
def test_a_staged_success_is_done_at_once_and_holds_its_workflow_open(
    spec: str,
) -> None:
    setup = _dispatched(spec)

    setup.succeed([setup.tree(stored=False)])

    assert setup.record(setup.train).status == TaskStatus.DONE
    assert setup.workers.answers(setup.train) == [UploadDisposition.PENDING]
    setup.finish_use()
    assert setup.record(setup.use).status == TaskStatus.DONE
    assert setup.status() == (WorkflowStatus.PENDING, 1, None)
    assert not setup.settled()


@pytest.mark.parametrize("spec", [_V1, _V2], ids=["v1", "v2"])
def test_the_owners_receipt_settles_it_durable_and_closes_the_workflow(
    spec: str,
) -> None:
    setup = _dispatched(spec)
    setup.succeed([setup.tree(stored=False)])
    setup.finish_use()

    setup.report(UploadReport.UPLOADED)

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.state is ArtifactDurability.DURABLE
    assert setup.workers.answers(setup.train)[-1] is UploadDisposition.DURABLE
    assert setup.status() == (WorkflowStatus.DONE, 0, None)
    settlement = setup.runtime.workflow_settlement(setup.workflow_id)
    assert settlement.settled and settlement.finished_ts == upload.settled_ts
    uploads = workflow_uploads_key(setup.workflow_id)
    assert not raw_control(setup.registry).exists(uploads)


def test_a_report_from_another_registration_or_dispatch_settles_nothing() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=False)])

    setup.report(UploadReport.UPLOADED, worker_id="wkr-2")
    setup.report(UploadReport.UPLOADED, dispatch_id="dsp-0")

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.pending
    assert setup.workers.answers(setup.train)[-1] is UploadDisposition.REJECTED


def test_a_receipt_ahead_of_its_success_settles_it_at_acceptance() -> None:
    setup = _dispatched()

    setup.report(UploadReport.UPLOADED)
    assert setup.record(setup.train).status == TaskStatus.DISPATCHED
    setup.succeed([setup.tree(stored=False)])

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.state is ArtifactDurability.DURABLE
    assert setup.workers.answers(setup.train) == [UploadDisposition.DURABLE]


def test_a_success_reported_again_hears_the_upload_owed_again() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=False)])

    setup.succeed([setup.tree(stored=False)])

    assert setup.workers.answers(setup.train) == [UploadDisposition.PENDING] * 2


def test_a_success_settling_a_cancel_owes_nothing_and_says_so() -> None:
    setup = _dispatched()
    setup.runtime.cancel_workflow(setup.workflow_id)

    setup.succeed([setup.tree(stored=False)])

    assert setup.record(setup.train).artifact_upload is None
    assert setup.workers.answers(setup.train) == [UploadDisposition.REJECTED]


def test_an_owner_that_left_after_storing_everything_settles_durable() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=True)])
    setup.finish_use()
    setup.workers.registered.discard("wkr-1")

    setup.runtime.artifact_owner_lost("wkr-1")
    setup.runtime._reconciler.check(setup.train)

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.state is ArtifactDurability.DURABLE
    assert setup.status() == (WorkflowStatus.DONE, 0, None)


@pytest.mark.parametrize("spec", [_V1, _V2], ids=["v1", "v2"])
def test_an_owner_that_left_before_storing_loses_only_what_is_missing(
    spec: str,
) -> None:
    setup = _dispatched(spec)
    stored = setup.file(b"metrics", "metrics.json", stored=True)
    setup.succeed([setup.tree(stored=False), stored])
    setup.finish_use()

    setup.runtime.artifact_owner_lost("wkr-1")
    setup.runtime._reconciler.check(setup.train)

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.state is ArtifactDurability.LOST
    tree, metrics = (
        setup.record(setup.train).artifacts[("final", name)]
        for name in ("final_model", "metrics.json")
    )
    assert upload.durability(tree)[0] is ArtifactDurability.LOST
    assert upload.durability(metrics)[0] is ArtifactDurability.DURABLE
    status, pending, failure = setup.status()
    assert (status, pending) == (WorkflowStatus.FAILED, 0)
    assert failure is not None and "artifact_lost" in failure
    assert setup.record(setup.train).status == TaskStatus.DONE


def test_a_store_that_cannot_be_asked_leaves_the_upload_owed() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=False)])
    setup.reader.store.stored_size = _unreachable  # type: ignore[method-assign]

    setup.runtime._reconciler.check(setup.train)

    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.pending
    assert not setup.settled()


def _unreachable(reference: ContentReference) -> int:
    raise ContentUnavailable("store down")


def test_store_access_renews_for_the_dispatch_owing_the_upload_only() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=False)])
    renew = setup.runtime.renewable_content_scope

    assert renew(setup.train, "wkr-1", "dsp-1") == "org"
    assert renew(setup.train, "wkr-2", "dsp-1") is None
    assert renew(setup.train, "wkr-1", "dsp-0") is None
    assert renew(setup.train, "wkr-1", None) is None

    setup.report(UploadReport.UPLOADED)
    assert renew(setup.train, "wkr-1", "dsp-1") is None


def test_the_owed_upload_is_stored_with_the_task_and_survives_a_restart() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=False)])
    setup.finish_use()
    control = raw_control(setup.registry)
    blob = control.get(task_state_key(setup.train))
    sources = control.hgetall(workflow_sources_key(setup.workflow_id))
    assert load_task_state(blob, sources).record.artifact_upload is not None

    restarted = _runtime(setup.registry, setup.workers, setup.reader)
    asyncio.run(restarted.rehydrate([setup.workflow_id]))

    assert not restarted.workflow_settlement(setup.workflow_id).settled
    assert restarted.artifact_upload_owners() == {"wkr-1"}
    restarted.record_artifact_upload(
        setup.train, "wkr-1", "dsp-1", UploadReport.UPLOADED
    )
    assert restarted.workflow_settlement(setup.workflow_id).settled


def test_a_restart_finding_the_owner_gone_settles_its_upload_from_the_store() -> None:
    setup = _dispatched()
    setup.succeed([setup.tree(stored=True)])
    setup.finish_use()
    setup.workers.registered.discard("wkr-1")

    restarted = _runtime(setup.registry, setup.workers, setup.reader)
    asyncio.run(restarted.rehydrate([setup.workflow_id]))
    restarted._reconciler.check(setup.train)

    assert restarted.workflow_settlement(setup.workflow_id).settled


_SIBLINGS = """apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: siblings}
spec:
  graph:
    nodes:
""" + "".join(
    f"""      - name: {name}
        spec:
          taskType: inference
          model: {{source: {{identifier: m}}}}
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [{name}]}}
""" for name in ("a", "b", "c")
)


def test_a_merged_dispatch_owes_each_settled_childs_upload_and_disowns_the_rest() -> (
    None
):
    setup = _Setup()
    runtime = setup.runtime
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _SIBLINGS, format="native")
    )
    a, b, c = (r.task_id for r in results)
    with runtime._cv:
        assert {runtime._ready.pop_ready_locked() for _ in range(2)} == {setup.train, a}
    assert runtime.plan_merge(a, 8, "wkr-1") == [b, c]
    record_dispatch(runtime, a, dispatch_id="dsp-m")
    staged = setup.file(b"weights", "model.bin", stored=False)
    payload = result_payload(setup.reader, a, {"value": a}, "org")
    payload["artifacts"] = [staged.model_dump(mode="json")]
    payload["child_result_references"] = {
        b: store_result(setup.reader, b, {"value": b}, "org").model_dump(mode="json")
    }
    payload["child_artifacts"] = {
        child: [staged.model_dump(mode="json")] for child in (b, c)
    }
    payload[ARTIFACTS_PENDING] = True

    runtime.mark_succeeded(a, "wkr-1", payload, _TS, dispatch_id="dsp-m")

    for owed in (a, b):
        upload = setup.record(owed).artifact_upload
        assert upload is not None and upload.dispatch_id == "dsp-m"
        assert setup.workers.answers(owed) == [UploadDisposition.PENDING]
    assert setup.record(c).status == TaskStatus.PENDING
    assert setup.workers.answers(c) == [UploadDisposition.REJECTED]
    workflow = setup.registry.get_workflow(workflow_id)
    assert workflow is not None and workflow.pending_artifact_uploads == 2

    runtime.record_artifact_upload(c, "wkr-1", "dsp-m", UploadReport.UPLOADED)
    assert setup.workers.answers(c)[-1] is UploadDisposition.REJECTED
    runtime.record_artifact_upload(b, "wkr-1", "dsp-m", UploadReport.UPLOADED)
    assert setup.workers.answers(b)[-1] is UploadDisposition.DURABLE


@pytest.mark.parametrize("spec", [_V1, _V2], ids=["v1", "v2"])
def test_a_cancelled_workflow_finishes_the_uploads_it_owes_before_it_cancels(
    spec: str,
) -> None:
    setup = _dispatched(spec)
    setup.succeed([setup.tree(stored=False)])

    setup.runtime.cancel_workflow(setup.workflow_id)

    assert setup.record(setup.use).status == TaskStatus.CANCELLED
    assert setup.status() == (WorkflowStatus.PENDING, 1, None)
    assert not setup.settled()
    assert setup.runtime.renewable_content_scope(setup.train, "wkr-1", "dsp-1")

    setup.report(UploadReport.UPLOADED)

    assert setup.status() == (WorkflowStatus.CANCELLED, 0, None)
    assert setup.settled()


def test_an_owner_lost_in_service_has_its_upload_settled_without_being_asked() -> None:
    setup = _Setup(reconciler=UploadReconciler)
    record_dispatch(setup.runtime, setup.train, dispatch_id="dsp-1")
    setup.succeed([setup.tree(stored=False)])

    setup.runtime.artifact_owner_lost("wkr-1")

    def owed() -> bool:
        upload = setup.record(setup.train).artifact_upload
        return upload is not None and upload.pending

    deadline = time.monotonic() + 5
    while owed():
        assert time.monotonic() < deadline, "the owner's loss was never reconciled"
        time.sleep(0.02)
    upload = setup.record(setup.train).artifact_upload
    assert upload is not None and upload.state is ArtifactDurability.LOST
    setup.runtime.shutdown()


def test_a_task_loading_a_lost_artifact_is_refused_at_dispatch() -> None:
    setup = _dispatched()
    stored = setup.file(b"metrics", "metrics.json", stored=True)
    setup.succeed([setup.tree(stored=False), stored])
    bindings = setup.record(setup.train).artifacts
    tree, metrics = (
        bindings[("final", "final_model")],
        bindings[("final", "metrics.json")],
    )
    assert setup.runtime.lost_artifact_reads([tree]) is None

    setup.runtime.artifact_owner_lost("wkr-1")
    setup.runtime._reconciler.check(setup.train)

    reason = setup.runtime.lost_artifact_reads([tree, metrics])
    assert reason is not None and reason.startswith("artifact_lost")
    assert setup.runtime.lost_artifact_reads([metrics]) is None

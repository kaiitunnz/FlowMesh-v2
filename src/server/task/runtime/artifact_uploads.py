"""The artifact uploads tasks' successes left owed, and settling one from the store.

Each owed upload lives on its task's record, which is durable; this keeps only what
reads it fast — which workflows each holds open, which worker owes it, and when each
workflow's last one settled — rebuilt from the records whenever one changes, so it
never counts on its own.

An upload whose owner can no longer finish it is settled from the store: control
checks every object the accepted artifacts name, a tree's members through its stored
manifest, by size and never by reading a payload. Every object present settles it
durable; an object the store says is absent loses the artifact naming it; a store that
cannot be asked leaves it owed, to be checked again later.
"""

import heapq
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from shared.content import ContentReference
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind

from ..artifact_durability import ArtifactLoss, ArtifactUpload, binding_key
from ..models import TaskRecord
from ..results import ResultReader, ResultUnavailable, ResultUnreadable

# How long an upload whose store could not be asked waits to be checked again.
RECONCILE_RETRY_SEC = 30.0
# How often a worker owing uploads, heard from, has the tasks waiting on them placed
# again, to read from it before the uploads land.
AWAITING_RETRY_SEC = 60.0


class ArtifactUploads:
    """The owed uploads by workflow and by owning worker, read from task records."""

    def __init__(self) -> None:
        self._pending: dict[str, set[str]] = defaultdict(set)
        self._owned: dict[str, set[str]] = defaultdict(set)
        self._by_dispatch: dict[str, set[str]] = defaultdict(set)
        self._owners: dict[str, tuple[str, str, str]] = {}
        self._last_settled: dict[str, float] = {}
        self._retried: dict[str, float] = {}

    def track_locked(self, record: TaskRecord) -> None:
        """Index a record's upload as it now stands."""
        task_id = record.task_id
        self._untrack(task_id)
        if (upload := record.artifact_upload) is None:
            return
        if upload.pending:
            self._pending[record.workflow_id].add(task_id)
            self._owned[upload.owner_id].add(task_id)
            self._by_dispatch[upload.dispatch_id].add(task_id)
            self._owners[task_id] = (
                record.workflow_id,
                upload.owner_id,
                upload.dispatch_id,
            )
        elif upload.settled_ts is not None:
            previous = self._last_settled.get(record.workflow_id, 0.0)
            self._last_settled[record.workflow_id] = max(previous, upload.settled_ts)

    def _untrack(self, task_id: str) -> None:
        if (owner := self._owners.pop(task_id, None)) is not None:
            workflow_id, worker_id, dispatch_id = owner
            _discard(self._pending, workflow_id, task_id)
            _discard(self._owned, worker_id, task_id)
            _discard(self._by_dispatch, dispatch_id, task_id)

    def owed_for_dispatch_locked(self, dispatch_id: str | None) -> set[str]:
        """The tasks whose upload a dispatch still owes."""
        return set(self._by_dispatch.get(dispatch_id or "", ()))

    def pending_locked(self, workflow_id: str) -> int:
        """How many of a workflow's tasks still owe their upload."""
        return len(self._pending.get(workflow_id, ()))

    def owed_by_locked(self, worker_id: str) -> list[str]:
        """The tasks whose upload a worker registration owes."""
        return sorted(self._owned.get(worker_id, ()))

    def retry_due_locked(self, worker_id: str, now: float) -> bool:
        """Whether the tasks waiting on a heard worker's uploads are due another try,
        noting the try."""
        if worker_id not in self._owned:
            self._retried.pop(worker_id, None)
            return False
        last = self._retried.setdefault(worker_id, now)
        if now - last < AWAITING_RETRY_SEC:
            return False
        self._retried[worker_id] = now
        return True

    def owners_locked(self) -> set[str]:
        """Every worker registration owing an upload."""
        return set(self._owned)

    def last_settled_locked(self, workflow_id: str) -> float | None:
        """When the workflow's last upload settled, if any did."""
        return self._last_settled.get(workflow_id)

    def forget_workflow_locked(self, workflow_id: str, task_ids: Iterable[str]) -> None:
        for task_id in task_ids:
            self._untrack(task_id)
        self._pending.pop(workflow_id, None)
        self._last_settled.pop(workflow_id, None)


def _discard(index: dict[str, set[str]], key: str, task_id: str) -> None:
    if (members := index.get(key)) is not None:
        members.discard(task_id)
        if not members:
            del index[key]


@dataclass(frozen=True)
class UploadCheck:
    """An owed upload to settle from the store: the upload as it stood, and the
    accepted artifacts it covers."""

    task_id: str
    upload: ArtifactUpload
    artifacts: tuple[ArtifactBinding, ...]


class StoreUncertain(Exception):
    """The store could not say whether an object is there."""


def probe_upload(
    results: ResultReader, artifacts: Iterable[ArtifactBinding]
) -> dict[str, ArtifactLoss]:
    """The accepted artifacts missing from the store, each with why; raises
    ``StoreUncertain`` when the store could not say.

    An archive views a tree and stands with it, so it is not checked on its own.
    """
    lost: dict[str, ArtifactLoss] = {}
    for binding in artifacts:
        if binding.kind is ArtifactKind.ARCHIVE:
            continue
        try:
            for member in _objects(results, binding):
                results.verify_stored(member)
        except ResultUnavailable as exc:
            raise StoreUncertain(str(exc)) from exc
        except ResultUnreadable as exc:
            lost[binding_key(binding)] = ArtifactLoss(
                message=f"{binding.name} at {binding.revision} is not in the shared "
                f"store and its producer can no longer upload it: {exc}"
            )
    for binding in artifacts:
        if binding.of is not None and binding_key(binding, binding.of) in lost:
            lost[binding_key(binding)] = lost[binding_key(binding, binding.of)]
    return lost


def _objects(
    results: ResultReader, binding: ArtifactBinding
) -> Iterable[ContentReference]:
    """Every object an artifact's bytes are in: a file's own, or a tree's members."""
    if binding.kind is not ArtifactKind.TREE:
        return (binding.reference,)
    scope = binding.reference.authorization_scope
    return tuple(
        member.reference(scope)
        for member in results.read_manifest(binding.reference).members
    )


class UploadReconciler:
    """Settles owed uploads from the store, one at a time, off the runtime lock.

    ``take`` reads an upload to check under the lock, or None once it no longer needs
    one; ``settle`` applies what the store showed under the lock, unless the upload
    moved on meanwhile. A store that could not be asked is asked again after
    ``RECONCILE_RETRY_SEC``.
    """

    def __init__(
        self,
        results: ResultReader,
        take: Callable[[str], UploadCheck | None],
        settle: Callable[[UploadCheck, dict[str, ArtifactLoss]], None],
        logger: logging.Logger,
        retry_sec: float = RECONCILE_RETRY_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._results = results
        self._take = take
        self._settle = settle
        self._logger = logger
        self._retry_sec = retry_sec
        self._clock = clock
        self._cv = threading.Condition()
        self._due: list[tuple[float, str]] = []
        self._queued: set[str] = set()
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="artifact-upload-reconcile", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()

    def request(self, *task_ids: str, delay: float = 0.0) -> None:
        """Check these tasks' uploads against the store."""
        at = self._clock() + delay
        with self._cv:
            for task_id in task_ids:
                if task_id in self._queued:
                    continue
                self._queued.add(task_id)
                heapq.heappush(self._due, (at, task_id))
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stopped and (
                    not self._due or self._due[0][0] > self._clock()
                ):
                    wait = self._due[0][0] - self._clock() if self._due else None
                    self._cv.wait(wait)
                if self._stopped:
                    return
                _, task_id = heapq.heappop(self._due)
                self._queued.discard(task_id)
            self.check(task_id)

    def check(self, task_id: str) -> None:
        """Settle one task's upload from the store, or ask again later."""
        try:
            check = self._take(task_id)
            if check is None:
                return
            lost = probe_upload(self._results, check.artifacts)
            self._settle(check, lost)
        except StoreUncertain as exc:
            self._logger.warning(
                "Upload of %s stays owed: the store could not be asked: %s",
                task_id,
                exc,
            )
            self.request(task_id, delay=self._retry_sec)
        except Exception:
            self._logger.exception("Checking the upload of %s failed", task_id)
            self.request(task_id, delay=self._retry_sec)

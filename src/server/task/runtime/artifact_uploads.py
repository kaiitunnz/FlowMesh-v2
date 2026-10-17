"""Owed artifact uploads, and settling one from the store.

Each owed upload lives on its task's record; the index here keeps which workflows each
holds open, which worker owes it, and when each workflow's last one settled, derived
from the record whenever it changes.

An upload whose owner can no longer finish it settles from the store, checked by each
object's size and a tree's stored manifest, never by reading a payload: every object
present settles it durable, an absent one loses the artifact naming it, and a store
that cannot be asked leaves it owed.
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
from shared.utils.recent import RecentSet

from ..artifact_durability import ArtifactLoss, ArtifactUpload, binding_key
from ..models import TaskRecord
from ..results import ResultReader, ResultUnavailable, ResultUnreadable

# How long an upload whose store could not be asked waits to be checked again.
RECONCILE_RETRY_SEC = 30.0
# How many ended registrations are remembered, for a success one reported before it
# ended that arrives after.
_DEPARTED_KEPT = 4096


class ArtifactUploads:
    """The owed uploads by workflow and by owning worker, read from task records."""

    def __init__(self) -> None:
        self._pending: dict[str, set[str]] = defaultdict(set)
        self._owned: dict[str, set[str]] = defaultdict(set)
        self._by_dispatch: dict[str, set[str]] = defaultdict(set)
        self._owners: dict[str, tuple[str, str, str]] = {}
        self._last_settled: dict[str, float] = {}
        self._lost_by_dispatch: dict[str, set[str]] = defaultdict(set)
        self._lost_of: dict[str, str] = {}
        # The uploads owed when their owner was declared lost, which only the store
        # settles; an upload accepted from that owner later is its own.
        self._fenced: set[str] = set()
        # The registrations known to have ended, whose successes arriving later owe
        # uploads they can no longer make.
        self._departed: RecentSet[str] = RecentSet(_DEPARTED_KEPT)
        # When each owner was last heard from.
        self._heard: dict[str, float] = {}

    def track_locked(self, record: TaskRecord) -> None:
        """Index a record's upload as it now stands."""
        task_id = record.task_id
        self._untrack(task_id)
        if (upload := record.internal.artifact_upload) is None:
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
            if upload.lost:
                self._lost_by_dispatch[upload.dispatch_id].add(task_id)
                self._lost_of[task_id] = upload.dispatch_id

    def _untrack(self, task_id: str) -> None:
        self._fenced.discard(task_id)
        if (dispatch_id := self._lost_of.pop(task_id, None)) is not None:
            _discard(self._lost_by_dispatch, dispatch_id, task_id)
        if (owner := self._owners.pop(task_id, None)) is not None:
            workflow_id, worker_id, dispatch_id = owner
            _discard(self._pending, workflow_id, task_id)
            _discard(self._owned, worker_id, task_id)
            _discard(self._by_dispatch, dispatch_id, task_id)
            if worker_id not in self._owned:
                self._heard.pop(worker_id, None)

    def owed_for_dispatch_locked(self, dispatch_id: str | None) -> set[str]:
        """Return the tasks whose upload a dispatch still owes."""
        return set(self._by_dispatch.get(dispatch_id or "", ()))

    def lost_for_dispatch_locked(self, dispatch_id: str | None) -> set[str]:
        """Return the tasks whose upload a dispatch owed settled with artifacts
        lost."""
        return set(self._lost_by_dispatch.get(dispatch_id or "", ()))

    def pending_tasks_locked(self) -> list[str]:
        """Return every task whose upload is still owed."""
        return list(self._owners)

    def pending_locked(self, workflow_id: str) -> int:
        """Return how many of a workflow's tasks still owe their upload."""
        return len(self._pending.get(workflow_id, ()))

    def owed_by_locked(self, worker_id: str) -> list[str]:
        """Return the tasks whose upload a worker registration owes."""
        return sorted(self._owned.get(worker_id, ()))

    def owes_locked(self, worker_id: str) -> bool:
        """Return whether a worker registration owes any upload."""
        return worker_id in self._owned

    def fence_locked(self, *task_ids: str) -> None:
        """Fence owed uploads whose owner was declared lost: it renews nothing for them
        and settles none of them."""
        self._fenced.update(task_id for task_id in task_ids if task_id in self._owners)

    def fenced_locked(self, task_id: str) -> bool:
        """Return whether an upload was owed when its owner was declared lost."""
        return task_id in self._fenced

    def depart_locked(self, worker_id: str) -> None:
        """Note a registration that ended."""
        self._departed.add(worker_id)

    def departed_locked(self, worker_id: str) -> bool:
        """Return whether a registration is known to have ended."""
        return worker_id in self._departed

    def heard_locked(self, worker_id: str, now: float, unheard_after: float) -> bool:
        """Note an owner heard from, and return whether it had gone unheard for
        ``unheard_after``."""
        if self._fenced.issuperset(self._owned.get(worker_id, ())):
            self._heard.pop(worker_id, None)
            return False
        last = self._heard.get(worker_id)
        self._heard[worker_id] = now
        return last is not None and now - last >= unheard_after

    def owners_locked(self) -> set[str]:
        """Return every worker registration owing an upload."""
        return set(self._owned)

    def last_settled_locked(self, workflow_id: str) -> float | None:
        """Return when the workflow's last upload settled, if any did."""
        return self._last_settled.get(workflow_id)

    def forget_workflow_locked(self, workflow_id: str, task_ids: Iterable[str]) -> None:
        """Drop a freed workflow's uploads."""
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


class ObjectsRetiring(StoreUncertain):
    """An object is being removed from the store, so whether it stays is not known
    until its removal ends."""


type Retiring = Callable[[list[ContentReference]], list[str]]


def probe_upload(
    results: ResultReader,
    artifacts: Iterable[ArtifactBinding],
    stored: set[ContentReference] | None,
    retiring: Retiring,
) -> dict[str, ArtifactLoss]:
    """Return the accepted artifacts missing from the store, each with why; raises
    ``StoreUncertain`` when the store could not say, and ``ObjectsRetiring`` when
    ``retiring`` finds an object being removed from it.

    An archive views a tree and stands with it, so it is not checked on its own.
    ``stored`` holds the objects an earlier probe found, which are not asked again,
    and gains each one this probe finds.
    """
    stored = set() if stored is None else stored
    lost: dict[str, ArtifactLoss] = {}
    for binding in artifacts:
        if binding.kind is ArtifactKind.ARCHIVE:
            continue
        try:
            objects = list(_objects(results, binding))
            if removing := retiring(objects):
                raise ObjectsRetiring(
                    f"{len(removing)} object(s) of {binding.name} are being "
                    f"removed from the store: {', '.join(removing[:5])}"
                )
            for member in objects:
                if member in stored:
                    continue
                results.verify_stored(member)
                stored.add(member)
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
    """Return every object an artifact's bytes are in: a file's own, or a tree's
    members."""
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
    ``RECONCILE_RETRY_SEC``, for the objects it did not find yet.
    """

    def __init__(
        self,
        results: ResultReader,
        take: Callable[[str], UploadCheck | None],
        settle: Callable[[UploadCheck, dict[str, ArtifactLoss]], None],
        retiring: Retiring,
        logger: logging.Logger,
        retry_sec: float = RECONCILE_RETRY_SEC,
        clock: Callable[[], float] = time.monotonic,
        run_thread: bool = True,
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
        self._stored: dict[str, set[ContentReference]] = {}
        self._stopped = False
        self._run_thread = run_thread
        self._thread: threading.Thread | None = None
        self._retiring = retiring

    def _ensure_thread(self) -> None:
        if not self._run_thread or self._thread is not None:
            return
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
            self._ensure_thread()
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
                self._stored.pop(task_id, None)
                return
            stored = self._stored.setdefault(task_id, set())
            lost = probe_upload(self._results, check.artifacts, stored, self._retiring)
            self._settle(check, lost)
            self._stored.pop(task_id, None)
        except ObjectsRetiring as exc:
            self._logger.warning("Upload of %s stays owed: %s", task_id, exc)
            self.request(task_id, delay=self._retry_sec)
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


def manual_reconciler(
    results: ResultReader,
    take: Callable[[str], UploadCheck | None],
    settle: Callable[[UploadCheck, dict[str, ArtifactLoss]], None],
    retiring: Retiring,
    logger: logging.Logger,
) -> UploadReconciler:
    """A reconciler its owner drives itself, which starts no thread."""
    return UploadReconciler(results, take, settle, retiring, logger, run_thread=False)

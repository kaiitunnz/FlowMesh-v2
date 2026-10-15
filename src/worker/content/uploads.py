"""Uploading the artifacts a task's success announced, after the success.

A dispatch stages its final artifacts into custody and reports success; the uploader
then puts each staged object in the shared store under the access its owing dispatch
renews, and reports when a task's whole set is there. Every report is repeated on a
pace until control answers it — the success until control accepts it, the upload until
control settles it — because the event helpers drop what they cannot send. Custody ends
when control settles every task of the dispatch.

Only the registration that staged a dispatch uploads it. Custody another registration
left on the node, this worker's earlier one or another worker's, is evidence for
control: it is reported as abandoned, and released once control says the upload settled.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference, ContentStoreError
from shared.tasks.artifact_upload import UploadDisposition, UploadReport, UploadSettled

from .custody import ArtifactCustody, StagedTask, UploadJournal

# How often an unanswered success or upload report is sent again.
RESEND_SEC = 30.0
# How often evidence of another registration's upload is reported while it is owed.
EVIDENCE_RESEND_SEC = 300.0
# How often the node's custody is scanned for evidence other registrations left.
EVIDENCE_SCAN_SEC = 600.0
# The pace of retried uploads while the store cannot take them, and of the loop after
# an unexpected failure.
_RETRY_MIN_SEC = 1.0
_RETRY_MAX_SEC = 60.0

type Identity = Callable[[], tuple[str, int]]
type Upload = Callable[[str, str, Path, str], ContentReference]
type ResendSuccess = Callable[[str, str, dict[str, Any]], None]
type ReportUpload = Callable[[str, str, UploadReport, str | None], None]


@dataclass
class _Publication:
    journal: UploadJournal
    # Whether this registration staged it and so uploads it.
    owned: bool
    # The staged objects not yet stored, each once, and the tasks each belongs to.
    queue: deque[ContentReference] = field(default_factory=deque)
    owners: dict[str, set[str]] = field(default_factory=dict)
    # What each task still has to store, by digest.
    remaining: dict[str, set[str]] = field(default_factory=dict)
    retry_at: float = 0.0
    backoff: float = _RETRY_MIN_SEC
    # When each unanswered report is next due, by task; the success under "".
    due: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, journal: UploadJournal, owned: bool) -> "_Publication":
        publication = cls(journal, owned)
        for task in journal.owes():
            if task.uploaded or task.failed is not None:
                continue
            digests = {member.content_digest for member in task.members}
            publication.remaining[task.task_id] = digests
            for member in task.members:
                digest = member.content_digest
                if digest not in publication.owners:
                    publication.owners[digest] = set()
                    publication.queue.append(member)
                publication.owners[digest].add(task.task_id)
        return publication

    def accepted(self) -> bool:
        return any(
            task.accepted or task.settled for task in self.journal.tasks.values()
        )

    def uploading(self) -> bool:
        return self.owned and bool(self.queue)


class ArtifactUploader:
    """Owes the shared store each artifact this worker staged, until control settles
    it."""

    def __init__(
        self,
        custody: ArtifactCustody,
        alias: str,
        identity: Identity,
        upload: Upload,
        resend_success: ResendSuccess,
        report: ReportUpload,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._custody = custody
        self._alias = alias
        self._identity = identity
        self._upload = upload
        self._resend_success = resend_success
        self._report = report
        self._logger = logger or logging.getLogger("artifact-uploads")
        self._clock = clock
        self._cv = threading.Condition()
        self._publications: dict[str, _Publication] = {}
        self._scan_at = 0.0
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Take up the custody other registrations left on the node, as evidence, and
        start uploading."""
        with self._cv:
            self._scan_evidence()
        self._thread = threading.Thread(
            target=self._run, name="artifact-uploads", daemon=True
        )
        self._thread.start()

    def begin(self, dispatch_id: str, scope: str) -> UploadJournal:
        """Open custody for a dispatch about to stage its final artifacts; nothing is
        written until it stages one."""
        worker_id, incarnation = self._identity()
        return UploadJournal(
            dispatch_id=dispatch_id,
            scope=scope,
            alias=self._alias,
            worker_id=worker_id,
            incarnation=incarnation,
            task_id="",
        )

    def stage(
        self,
        journal: UploadJournal,
        task_id: str,
        path: Path,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Copy one of a task's artifact files into the dispatch's custody."""
        if (staged := journal.tasks.get(task_id)) is None:
            staged = journal.tasks[task_id] = StagedTask(task_id=task_id)
            # Recorded before its first file, so control can be asked about the task.
            self._custody.record(journal)
        reference = self._custody.stage(
            journal.scope, journal.dispatch_id, path, cancelled=cancelled
        )
        staged.members.append(reference)
        return reference

    def abandon(self, journal: UploadJournal) -> None:
        """End custody for a dispatch whose success will not report it."""
        self._custody.release(journal.scope, journal.dispatch_id)

    def submit(
        self, journal: UploadJournal, task_id: str, success: dict[str, Any]
    ) -> None:
        """Owe the uploads of a dispatch whose success is about to be reported; its
        custody ends if that cannot be recorded."""
        journal.task_id = task_id
        journal.success = success
        try:
            self._custody.record(journal)
        except BaseException:
            self.abandon(journal)
            raise
        with self._cv:
            publication = _Publication.of(journal, owned=True)
            publication.due[""] = self._clock() + RESEND_SEC
            self._publications[journal.dispatch_id] = publication
            self._cv.notify_all()

    def dispatch_of(self, task_id: str) -> str | None:
        """Return the dispatch whose upload this registration still owes, by any task
        it covers or the task it ran, whose access the uploads are written under."""
        with self._cv:
            for dispatch_id, publication in self._publications.items():
                journal = publication.journal
                if not publication.owned or not journal.owes():
                    continue
                task = journal.tasks.get(task_id)
                if journal.task_id == task_id or (
                    task is not None and not task.settled
                ):
                    return dispatch_id
        return None

    def settle(self, settled: UploadSettled) -> None:
        """Take control's answer about one task's upload.

        A pending answer accepts the success; a report the task already has to make is
        then made again at once, since the one before it may have been lost.
        """
        with self._cv:
            publication = self._publications.get(settled.dispatch_id)
            if publication is None:
                return
            journal = publication.journal
            if (task := journal.tasks.get(settled.task_id)) is None:
                return
            task.accepted = True
            publication.due.pop("", None)
            if settled.disposition is UploadDisposition.PENDING:
                if not publication.owned:
                    publication.due[task.task_id] = self._clock() + EVIDENCE_RESEND_SEC
                elif task.uploaded or task.failed is not None:
                    publication.due[task.task_id] = self._clock()
                self._cv.notify_all()
                return
            task.settled = True
            publication.due.pop(task.task_id, None)
            if not publication.owned and not self._caught_up(publication):
                return
            if journal.owes():
                if publication.owned:
                    self._record(publication)
                return
            del self._publications[settled.dispatch_id]
            self._cv.notify_all()
        self._custody.release(journal.scope, journal.dispatch_id)

    def _caught_up(self, publication: _Publication) -> bool:
        """Take up the tasks another registration's journal gained since it was read,
        and return whether its success was reported, so a settled answer may end its
        custody."""
        journal = publication.journal
        try:
            current = self._custody.journal(journal.scope, journal.dispatch_id)
        except (OSError, ValidationError):
            return False
        if current is None:
            return True
        journal.task_id = current.task_id
        for task_id, task in current.tasks.items():
            if task_id not in journal.tasks:
                journal.tasks[task_id] = task
                if not task.settled:
                    publication.due[task_id] = self._clock()
        return bool(journal.task_id)

    def rebind(self) -> None:
        """Stop uploading what an earlier registration staged; it is evidence now."""
        with self._cv:
            for publication in self._publications.values():
                if publication.owned:
                    publication.owned = False
                    publication.queue.clear()
                    publication.due = {
                        task.task_id: 0.0 for task in publication.journal.owes()
                    }
            self._cv.notify_all()

    def stop(self, timeout: float) -> None:
        """Finish the uploads this registration owes, within ``timeout``, then stop."""
        deadline = self._clock() + max(0.0, timeout)
        with self._cv:
            self._cv.wait_for(
                lambda: not any(p.uploading() for p in self._publications.values()),
                timeout=max(0.0, deadline - self._clock()),
            )
            self._stopping = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - self._clock()))

    def _run(self) -> None:
        backoff = _RETRY_MIN_SEC
        while True:
            try:
                if not self._step():
                    return
                backoff = _RETRY_MIN_SEC
            except Exception:
                self._logger.exception(
                    "Artifact uploads failed unexpectedly; resuming in %.0f s", backoff
                )
                with self._cv:
                    if self._stopping:
                        return
                    self._cv.wait(backoff)
                backoff = min(_RETRY_MAX_SEC, backoff * 2)

    def _step(self) -> bool:
        """Make the reports and the upload now due; False once stopping."""
        with self._cv:
            if self._stopping:
                return False
            if self._clock() >= self._scan_at:
                self._scan_evidence()
            sends = self._due_sends()
            work = self._next_upload()
            if not sends and work is None:
                self._cv.wait(self._wait_sec())
                return True
        for send in sends:
            try:
                send()
            except Exception:
                self._logger.debug("could not send an upload report", exc_info=True)
        if work is not None:
            self._upload_one(*work)
        return True

    def _scan_evidence(self) -> None:
        """Take up the custody other registrations on the node hold, to ask control
        about each upload they still owe; a dispatch this worker's earlier
        registration staged without reporting its success is released, and another's
        only once its success was reported and nothing it covers is owed."""
        self._scan_at = self._clock() + EVIDENCE_SCAN_SEC
        worker_id, _ = self._identity()
        for journal in self._custody.journals():
            if journal.dispatch_id in self._publications:
                continue
            if journal.alias == self._alias and journal.worker_id == worker_id:
                continue
            if journal.alias == self._alias and not journal.task_id:
                self._custody.release(journal.scope, journal.dispatch_id)
                continue
            if not journal.owes():
                if journal.task_id:
                    self._custody.release(journal.scope, journal.dispatch_id)
                continue
            publication = _Publication(journal, owned=False)
            publication.due = {task.task_id: 0.0 for task in journal.owes()}
            self._publications[journal.dispatch_id] = publication

    def _due_sends(self) -> list[Callable[[], None]]:
        now = self._clock()
        sends: list[Callable[[], None]] = []
        for publication in self._publications.values():
            journal = publication.journal
            for key, at in list(publication.due.items()):
                if at > now:
                    continue
                if key == "":
                    if publication.owned and not publication.accepted():
                        publication.due[key] = now + RESEND_SEC
                        sends.append(self._success_send(journal))
                    else:
                        del publication.due[key]
                    continue
                task = journal.tasks.get(key)
                if task is None or task.settled:
                    del publication.due[key]
                    continue
                publication.due[key] = now + (
                    RESEND_SEC if publication.owned else EVIDENCE_RESEND_SEC
                )
                sends.append(self._report_send(publication, task))
        return sends

    def _success_send(self, journal: UploadJournal) -> Callable[[], None]:
        return lambda: self._resend_success(
            journal.task_id, journal.dispatch_id, journal.success
        )

    def _report_send(
        self, publication: _Publication, task: StagedTask
    ) -> Callable[[], None]:
        dispatch_id = publication.journal.dispatch_id
        if not publication.owned:
            report, error = UploadReport.ABANDONED, None
        elif task.failed is not None:
            report, error = UploadReport.UNUPLOADABLE, task.failed
        else:
            report, error = UploadReport.UPLOADED, None
        return lambda: self._report(task.task_id, dispatch_id, report, error)

    def _next_upload(self) -> tuple[_Publication, ContentReference] | None:
        now = self._clock()
        for publication in self._publications.values():
            if publication.uploading() and publication.retry_at <= now:
                return publication, publication.queue[0]
        return None

    def _wait_sec(self) -> float:
        now = self._clock()
        due = [self._scan_at - now] + [
            at - now
            for publication in self._publications.values()
            for at in publication.due.values()
        ]
        due += [p.retry_at - now for p in self._publications.values() if p.uploading()]
        return max(0.05, min(due))

    def _upload_one(self, publication: _Publication, member: ContentReference) -> None:
        journal = publication.journal
        failure: str | None = None
        try:
            source = self._custody.source(journal, member)
            stored = self._upload(
                journal.task_id, journal.scope, source, member.media_type
            )
            if (stored.identity, stored.size_bytes) != (
                member.identity,
                member.size_bytes,
            ):
                failure = f"the custody copy of {member.content_digest} changed"
        except FileNotFoundError:
            failure = f"the custody copy of {member.content_digest} is gone"
        except Exception as exc:
            with self._cv:
                wait = publication.backoff
                publication.retry_at = self._clock() + wait
                publication.backoff = min(_RETRY_MAX_SEC, wait * 2)
            self._logger.warning(
                "Upload of %s for %s failed; retrying in %.0f s: %s",
                member.content_digest,
                journal.dispatch_id,
                wait,
                exc,
                exc_info=not isinstance(exc, (ContentStoreError, OSError)),
            )
            return
        with self._cv:
            if publication.queue and publication.queue[0] is member:
                publication.queue.popleft()
            publication.backoff = _RETRY_MIN_SEC
            finished: list[StagedTask] = []
            for task_id in publication.owners.pop(member.content_digest, ()):
                remaining = publication.remaining.get(task_id)
                if remaining is None:
                    continue
                remaining.discard(member.content_digest)
                task = journal.tasks[task_id]
                if not remaining:
                    del publication.remaining[task_id]
                if failure is not None and task.failed is None:
                    task.failed = failure
                    self._logger.error(
                        "Upload of task %s in dispatch %s cannot finish: %s",
                        task_id,
                        journal.dispatch_id,
                        failure,
                    )
                # A failed task is reported once its other files are stored too, so
                # control finds every one it can.
                if not remaining:
                    task.uploaded = task.failed is None
                    finished.append(task)
            if finished:
                self._record(publication)
                now = self._clock()
                for task in finished:
                    publication.due[task.task_id] = now
            self._cv.notify_all()

    def _record(self, publication: _Publication) -> None:
        """Write a publication's journal; custody that cannot be recorded cannot be
        trusted, so every task it still owes is reported unuploadable."""
        try:
            self._custody.record(publication.journal)
        except OSError as exc:
            failure = f"the upload's custody journal could not be written: {exc}"
            self._logger.error(
                "Upload of dispatch %s cannot finish: %s",
                publication.journal.dispatch_id,
                failure,
            )
            publication.queue.clear()
            publication.remaining.clear()
            publication.owners.clear()
            now = self._clock()
            for task in publication.journal.owes():
                if task.failed is None and not task.uploaded:
                    task.failed = failure
                publication.due[task.task_id] = now

"""Uploading the artifacts a task's success announced, after the success.

A dispatch stages its final artifacts into custody and reports success; the uploader
then puts each staged object in the shared store under the access its owing dispatch
renews, and reports when a task's whole set is there. Every report is repeated on a
pace until control answers it — the success until control accepts it, the upload until
control settles it — because the event helpers drop what they cannot send. Custody ends
when control settles every task of the dispatch.

Only the registration that staged a dispatch uploads it. Custody found from an earlier
registration of this worker is evidence for control, which settles it from the store:
it is reported as abandoned, and released once control answers.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.content import ContentReference, ContentStoreError
from shared.tasks.artifact_upload import UploadReport, UploadSettled

from .custody import ArtifactCustody, StagedTask, UploadJournal

# How often an unanswered success or upload report is sent again.
RESEND_SEC = 30.0
# The pace of retried uploads while the store cannot take them.
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
    stored: set[str] = field(default_factory=set)
    failed: str | None = None
    retry_at: float = 0.0
    backoff: float = _RETRY_MIN_SEC
    # When each unanswered report is next due, by task; the success under "".
    due: dict[str, float] = field(default_factory=dict)

    def accepted(self) -> bool:
        return any(
            task.accepted or task.settled for task in self.journal.tasks.values()
        )

    def uploading(self) -> bool:
        return (
            self.owned
            and self.failed is None
            and any(not task.uploaded for task in self.journal.owes())
        )


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
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def custody(self) -> ArtifactCustody:
        return self._custody

    def start(self) -> None:
        """Take up the custody earlier registrations of this worker left, as evidence,
        and start uploading."""
        with self._cv:
            for journal in self._custody.journals():
                if journal.alias != self._alias:
                    continue
                if not journal.owes() or not journal.task_id:
                    # Staged by a dispatch whose success was never reported.
                    self._custody.release(journal.scope, journal.dispatch_id)
                    continue
                self._publications[journal.dispatch_id] = _Publication(
                    journal, owned=False
                )
        self._thread = threading.Thread(
            target=self._run, name="artifact-uploads", daemon=True
        )
        self._thread.start()

    def begin(self, dispatch_id: str, scope: str) -> UploadJournal:
        """Open custody for a dispatch about to stage its final artifacts."""
        worker_id, incarnation = self._identity()
        journal = UploadJournal(
            dispatch_id=dispatch_id,
            scope=scope,
            alias=self._alias,
            worker_id=worker_id,
            incarnation=incarnation,
            task_id="",
        )
        self._custody.record(journal)
        return journal

    def stage(
        self,
        journal: UploadJournal,
        task_id: str,
        path: Path,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContentReference:
        """Copy one of a task's artifact files into the dispatch's custody."""
        reference = self._custody.stage(
            journal.scope, journal.dispatch_id, path, cancelled=cancelled
        )
        staged = journal.tasks.setdefault(task_id, StagedTask(task_id=task_id))
        staged.members.append(reference)
        return reference

    def abandon(self, journal: UploadJournal) -> None:
        """End custody for a dispatch whose success will not report it."""
        self._custody.release(journal.scope, journal.dispatch_id)

    def submit(
        self, journal: UploadJournal, task_id: str, success: dict[str, Any]
    ) -> None:
        """Owe the uploads of a dispatch whose success is about to be reported."""
        journal.task_id = task_id
        journal.success = success
        self._custody.record(journal)
        with self._cv:
            publication = _Publication(journal, owned=True)
            publication.due[""] = self._clock() + RESEND_SEC
            self._publications[journal.dispatch_id] = publication
            self._cv.notify_all()

    def dispatch_of(self, task_id: str) -> str | None:
        """The dispatch whose upload this registration still owes, by any task it
        covers or the task it ran, whose access the uploads are written under."""
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
        """Take control's answer about one task's upload."""
        with self._cv:
            publication = self._publications.get(settled.dispatch_id)
            if publication is None:
                return
            journal = publication.journal
            if (task := journal.tasks.get(settled.task_id)) is None:
                return
            if settled.disposition.settled:
                task.settled = True
            task.accepted = True
            publication.due.pop(settled.task_id, None)
            publication.due.pop("", None)
            if journal.owes():
                self._custody.record(journal)
                return
            del self._publications[settled.dispatch_id]
            self._cv.notify_all()
        self._custody.release(journal.scope, journal.dispatch_id)

    def rebind(self) -> None:
        """Stop uploading what an earlier registration staged; it is evidence now."""
        with self._cv:
            for publication in self._publications.values():
                publication.owned = False
                publication.due = {}
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
        while True:
            with self._cv:
                if self._stopping:
                    return
                sends = self._due_sends()
                work = self._next_upload()
                if not sends and work is None:
                    self._cv.wait(self._wait_sec())
                    continue
            for send in sends:
                try:
                    send()
                except Exception:
                    self._logger.debug("could not send an upload report", exc_info=True)
            if work is not None:
                self._upload_one(*work)

    def _due_sends(self) -> list[Callable[[], None]]:
        now = self._clock()
        sends: list[Callable[[], None]] = []
        for publication in self._publications.values():
            journal = publication.journal
            for key, at in list(publication.due.items()):
                if at > now:
                    continue
                publication.due[key] = now + RESEND_SEC
                if key == "":
                    if publication.owned and not publication.accepted():
                        sends.append(self._success_send(journal))
                    else:
                        del publication.due[key]
                    continue
                task = journal.tasks.get(key)
                if task is None or task.settled:
                    del publication.due[key]
                    continue
                sends.append(self._report_send(publication, key))
            if not publication.owned:
                for task in journal.owes():
                    publication.due.setdefault(task.task_id, now)
        return sends

    def _success_send(self, journal: UploadJournal) -> Callable[[], None]:
        return lambda: self._resend_success(
            journal.task_id, journal.dispatch_id, journal.success
        )

    def _report_send(
        self, publication: _Publication, task_id: str
    ) -> Callable[[], None]:
        journal = publication.journal
        if not publication.owned:
            report, error = UploadReport.ABANDONED, None
        elif publication.failed is not None:
            report, error = UploadReport.UNUPLOADABLE, publication.failed
        else:
            report, error = UploadReport.UPLOADED, None
        return lambda: self._report(task_id, journal.dispatch_id, report, error)

    def _next_upload(self) -> tuple[_Publication, ContentReference] | None:
        now = self._clock()
        for publication in self._publications.values():
            if not publication.uploading() or publication.retry_at > now:
                continue
            for digest, member in publication.journal.members().items():
                if digest not in publication.stored:
                    return publication, member
        return None

    def _wait_sec(self) -> float:
        now = self._clock()
        due = [
            at - now
            for publication in self._publications.values()
            for at in publication.due.values()
        ] + [p.retry_at - now for p in self._publications.values() if p.uploading()]
        return max(0.05, min(due, default=RESEND_SEC))

    def _upload_one(self, publication: _Publication, member: ContentReference) -> None:
        journal = publication.journal
        source = self._custody.source(journal, member)
        failure: str | None = None
        try:
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
        except (ContentStoreError, OSError) as exc:
            with self._cv:
                publication.retry_at = self._clock() + publication.backoff
                publication.backoff = min(_RETRY_MAX_SEC, publication.backoff * 2)
            self._logger.warning(
                "Upload of %s for %s failed; retrying in %.0f s: %s",
                member.content_digest,
                journal.dispatch_id,
                publication.backoff,
                exc,
            )
            return
        with self._cv:
            if failure is not None:
                self._logger.error(
                    "Upload of dispatch %s cannot finish: %s",
                    journal.dispatch_id,
                    failure,
                )
                publication.failed = failure
                self._report_tasks(publication, journal.owes())
                self._cv.notify_all()
                return
            publication.stored.add(member.content_digest)
            publication.backoff = _RETRY_MIN_SEC
            done = [
                task
                for task in journal.owes()
                if not task.uploaded
                and all(m.content_digest in publication.stored for m in task.members)
            ]
            for task in done:
                task.uploaded = True
            if done:
                self._custody.record(journal)
                self._report_tasks(publication, done)
            self._cv.notify_all()

    def _report_tasks(self, publication: _Publication, tasks: list[Any]) -> None:
        now = self._clock()
        for task in tasks:
            publication.due[task.task_id] = now

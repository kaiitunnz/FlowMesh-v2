"""Deciding who may hydrate which object, and arranging the transfer.

The control-plane actor over the holder directory. A worker that needs an object asks
here; the authority checks that the asking worker is the one running the task and that
the task's own binding names exactly this reference, resolves a live holder, mints one
grant, and hands it to both ends over their authenticated attachments. Then it steps
out: the bytes move worker to worker over the relay, and no payload passes through here.

It reads consumer bindings as evidence and never writes one. A denial creates nothing —
no invocation, no claim, no credit — and settles nothing: the asking worker surfaces a
typed hydration failure and its own consumer decides what that means.
"""

import logging
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from shared.content import ContentHydrationGrant, ContentReference
from shared.schemas.command import MediatedOpMessage
from shared.utils.ids import new_hydration_grant_id, new_relay_session_id

from ..registries.worker import Worker, WorkerRegistry
from .directory import ContentHolderDirectory, ContentHolderRecord
from .sessions import ContentTransferSessions

# Whether the task's own binding entitles this worker to read this exact reference.
BindingCheck = Callable[[str, str, ContentReference], bool]


class HydrationDenial(StrEnum):
    """Why control refused to authorize a hydration."""

    NO_BINDING = "no_binding"
    NOT_TRACKED = "not_tracked"
    HOLDER_UNAVAILABLE = "holder_unavailable"


class ContentHydrationAuthority:
    """Mints the grant one worker needs to read one object from another."""

    def __init__(
        self,
        directory: ContentHolderDirectory,
        worker_registry: WorkerRegistry,
        *,
        authorizes: BindingCheck,
        grant_ttl_sec: float,
        sessions: ContentTransferSessions,
        logger: logging.Logger | None = None,
    ) -> None:
        self._directory = directory
        self._workers = worker_registry
        self._authorizes = authorizes
        self._grant_ttl_sec = grant_ttl_sec
        self._sessions = sessions
        self._logger = logger or logging.getLogger("content-authority")

    def record_holding(self, worker_id: str, held: Sequence[tuple[str, str]]) -> None:
        """Note the objects a worker holds, from its write or its periodic report.

        A report is what keeps a record live, and it arrives under the reporting
        worker's current incarnation, so a restarted holder supersedes the records its
        previous one left behind rather than waiting for them to lapse.
        """
        worker = self._workers.get_worker(worker_id)
        if worker is None:
            return
        for scope, digest in held:
            self._directory.record(
                scope,
                digest,
                worker_id=worker_id,
                node_id=worker.node_id,
                generation=worker.incarnation,
            )

    def authorize(
        self, worker_id: str, task_id: str, reference: ContentReference
    ) -> None:
        """Grant one hydration, or relay the reason it is refused."""
        requester = self._workers.get_worker(worker_id)
        if requester is None:
            # Nothing to answer: a request from a worker the registry no longer knows
            # has no attachment to relay a grant or a refusal over.
            return
        if not self._authorizes(task_id, worker_id, reference):
            self._deny(requester, reference, HydrationDenial.NO_BINDING)
            return
        reported = self._directory.holders(
            reference.authorization_scope, reference.content_digest
        )
        if not reported:
            # Nothing ever reported holding it — no copy was ever announced, or every
            # holder's report has lapsed. Its consumer reads the shared store instead.
            self._deny(requester, reference, HydrationDenial.NOT_TRACKED)
            return
        holder = self._resolve_holder(reference, reported, exclude=worker_id)
        if holder is None:
            self._deny(requester, reference, HydrationDenial.HOLDER_UNAVAILABLE)
            return
        grant = ContentHydrationGrant(
            grant_id=new_hydration_grant_id(),
            reference=reference,
            requester_subject=worker_id,
            requester_origin_id=requester.node_id,
            holder_id=holder.id,
            holder_generation=holder.incarnation,
            transfer_session_id=new_relay_session_id(),
            expires_at_epoch=time.time() + self._grant_ttl_sec,
        )
        # The transfer's routing record: the relay bridges frames between these two
        # nodes, and their workers receive them. It carries no object identity.
        self._sessions.open(
            grant.transfer_session_id,
            origin_node=requester.node_id,
            target_node=holder.node_id,
            origin_worker=worker_id,
            target_worker=holder.id,
        )
        payload = {"grant": grant.model_dump(mode="json")}
        self._relay(holder, "content_serve_grant", payload)
        self._relay(requester, "content_grant", payload)

    def _resolve_holder(
        self,
        reference: ContentReference,
        reported: list[ContentHolderRecord],
        *,
        exclude: str,
    ) -> Worker | None:
        """A live worker that holds the object, or None. Evidence, not authority."""
        for record in reported:
            if record.worker_id == exclude:
                continue
            worker = self._workers.get_worker(record.worker_id)
            if worker is not None and worker.incarnation == record.generation:
                return worker
            # The reporting worker is gone or came back as another incarnation, so its
            # report describes content no one can serve. A restarted worker re-reports
            # what it still holds, which lands as a record for its new incarnation.
            self._directory.forget(
                reference.authorization_scope,
                reference.content_digest,
                record.worker_id,
            )
        return None

    def _deny(
        self, requester: Worker, reference: ContentReference, denial: HydrationDenial
    ) -> None:
        self._logger.info(
            "content hydration refused for %s: %s", requester.id, denial.value
        )
        self._relay(
            requester,
            "content_grant_denied",
            {"reference": reference.model_dump(mode="json"), "reason": denial.value},
        )

    def _relay(self, worker: Worker, frame_kind: str, payload: dict[str, Any]) -> None:
        self._workers.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker.id, frame_kind=frame_kind, payload=payload
            ),
        )

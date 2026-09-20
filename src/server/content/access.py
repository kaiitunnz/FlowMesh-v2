"""Issuing a task the access it needs to the shared content store.

Control decides what a worker may reach and for how long. For each dispatched task that
will touch content, the broker mints a `ContentStoreAccessGrant` naming the task, its
scope, the worker incarnation running it, and a short expiry, asks the backend for
session material cut to that scope, and relays the two together over the worker's
authenticated attachment. A later dispatch or a recovery gets fresh material; nothing
extends what was already issued.

The material is the backend's, not the fabric's: an S3-compatible store cuts a
short-lived session over the scope's prefix, and a deployment whose backend cannot cut
one says so rather than pretending. What the broker never does is widen the grant — no
list, no delete, no binding — or keep the material anywhere but the message that carries
it.
"""

import logging
import time
from typing import Protocol

from shared.content import (
    ContentOperationKind,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ScopedContentCredential,
)
from shared.schemas.command import MediatedOpMessage
from shared.utils.ids import new_store_access_grant_id

from ..registries.worker import Worker, WorkerRegistry

_OPERATIONS = (ContentOperationKind.READ, ContentOperationKind.WRITE)


class ScopedCredentialMinter(Protocol):
    """Cuts backend session material for one scope, under one grant's terms."""

    def mint(
        self, scope: str, operations: tuple[ContentOperationKind, ...], ttl_sec: float
    ) -> ScopedContentCredential: ...

    @property
    def policy_version(self) -> str:
        """The backend policy generation this minter cuts against."""


class ContentAccessBroker:
    """Mints and relays the access each dispatched task uses against the store."""

    def __init__(
        self,
        worker_registry: WorkerRegistry,
        minter: ScopedCredentialMinter,
        *,
        grant_ttl_sec: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self._workers = worker_registry
        self._minter = minter
        self._grant_ttl_sec = grant_ttl_sec
        self._logger = logger or logging.getLogger("content-access")

    def issue(self, worker_id: str, task_id: str, scope: str) -> None:
        """Give one task the access it will read and write its content under."""
        worker = self._workers.get_worker(worker_id)
        if worker is None:
            return
        grant = ContentStoreAccessGrant(
            grant_id=new_store_access_grant_id(),
            task_id=task_id,
            authorization_scope=scope,
            subject=worker_id,
            subject_generation=worker.incarnation,
            operations=_OPERATIONS,
            backend_policy_version=self._minter.policy_version,
            expires_at_epoch=time.time() + self._grant_ttl_sec,
        )
        try:
            credential = self._minter.mint(scope, _OPERATIONS, self._grant_ttl_sec)
        except Exception:
            # A task that cannot be given access runs without it and fails its first
            # content read, which is the same outcome as a store it cannot reach.
            self._logger.exception(
                "could not mint content store access for %s", task_id
            )
            return
        self._relay(worker, ContentStoreAccess(grant=grant, credential=credential))

    def _relay(self, worker: Worker, access: ContentStoreAccess) -> None:
        self._workers.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker.id,
                frame_kind="content_access",
                payload=access.model_dump(mode="json"),
            ),
        )

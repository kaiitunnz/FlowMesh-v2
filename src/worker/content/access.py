"""The access this worker holds against the shared content store.

Control relays one access per dispatched task and scope; the registry keeps it for as
long as it is good for and opens the store with it. A task with no live access does not
fall back to anything — reaching the store is exactly what it was not given — so its
first read or write fails rather than quietly running under whatever credential the
process happens to have. Control relays the access on its own path, so a task can reach
its first content operation while its access is still in flight; a read waits a bounded
moment for the one it was sent before deciding it has none.

The material never leaves here. It opens a backend client and is held only for as long
as the grant it came with; nothing writes it down, reports it, or renders it.
"""

import logging
import threading
import time

from shared.content import (
    BACKEND_FILESYSTEM,
    BACKEND_S3,
    ContentStoreAccess,
    ContentStoreError,
    FabricObjectStore,
    ObjectStoreConfig,
    ScopedContentCredential,
)
from shared.content.s3_store import S3ObjectStore

from .cas import SharedFilesystemObjectStore

_AccessKey = tuple[str, str]


class ContentAccessDenied(ContentStoreError):
    """This task holds no live access to the shared store in this scope."""


class ContentAccessRegistry:
    """The stores this worker may open, one per task and scope control granted."""

    def __init__(
        self,
        cfg: ObjectStoreConfig,
        logger: logging.Logger | None = None,
        *,
        arrival_wait_sec: float = 5.0,
    ) -> None:
        self._cfg = cfg
        self._logger = logger or logging.getLogger("content-access")
        self._arrival_wait_sec = arrival_wait_sec
        self._arrived = threading.Condition()
        self._granted: dict[_AccessKey, ContentStoreAccess] = {}
        self._stores: dict[_AccessKey, FabricObjectStore] = {}

    def accept(self, access: ContentStoreAccess) -> None:
        """Take one access control minted for a task of this worker."""
        key = (access.grant.task_id, access.grant.authorization_scope)
        with self._arrived:
            self._granted[key] = access
            self._stores.pop(key, None)
            self._expire()
            self._arrived.notify_all()

    def store_for(self, task_id: str, scope: str) -> FabricObjectStore:
        """The store this task opens for a scope, or a refusal if it holds none."""
        key = (task_id, scope)
        with self._arrived:
            access = self._await_access(key)
            if access is None:
                raise ContentAccessDenied(
                    f"task {task_id} holds no content store access in scope {scope}"
                )
            if access.grant.expired():
                self._forget(key)
                raise ContentAccessDenied(
                    f"the content store access for task {task_id} has expired"
                )
            if (store := self._stores.get(key)) is None:
                store = self._open(access.credential)
                self._stores[key] = store
            return store

    def _await_access(self, key: _AccessKey) -> ContentStoreAccess | None:
        """This task's access, waiting out a relay that has not landed yet."""
        if (access := self._granted.get(key)) is not None:
            return access
        self._arrived.wait_for(
            lambda: key in self._granted, timeout=self._arrival_wait_sec
        )
        return self._granted.get(key)

    def release(self, task_id: str) -> None:
        """Drop everything a finished task was given."""
        with self._arrived:
            for key in [k for k in self._granted if k[0] == task_id]:
                self._forget(key)

    def _open(self, credential: ScopedContentCredential) -> FabricObjectStore:
        match self._cfg.backend:
            case x if x == BACKEND_S3:
                return self._open_s3(credential)
            case x if x == BACKEND_FILESYSTEM:
                return SharedFilesystemObjectStore(self._cfg.filesystem_root)
            case _:
                raise ContentAccessDenied(
                    f"unknown content store backend {self._cfg.backend}"
                )

    def _open_s3(self, credential: ScopedContentCredential) -> FabricObjectStore:
        import boto3
        from botocore.client import Config

        material = credential.material
        client = boto3.client(
            "s3",
            endpoint_url=self._cfg.endpoint_url or None,
            aws_access_key_id=material.get("access_key") or None,
            aws_secret_access_key=material.get("secret_key") or None,
            aws_session_token=material.get("session_token") or None,
            region_name=self._cfg.region,
            config=Config(signature_version="s3v4"),
        )
        return S3ObjectStore(client, self._cfg.bucket, prefix=self._cfg.prefix)

    def _forget(self, key: _AccessKey) -> None:
        self._granted.pop(key, None)
        self._stores.pop(key, None)

    def _expire(self) -> None:
        now = time.time()
        for key, access in list(self._granted.items()):
            if access.grant.expired(now):
                self._forget(key)

"""The access this worker holds against the shared content store.

Control relays one access per dispatched task and scope; the registry keeps it for as
long as it is good for and opens the store with it. Its grant's expiry is what ends it,
not the task's own end: a boundary settles after the episode that raised it has already
returned, so the outcome it materializes is written under a task whose body is over.

A task with no live access does not fall back to anything — reaching the store is
exactly what it was not given — so its first read or write fails rather than quietly
running under whatever credential the process happens to have. Control relays the
access on its own path, so a task can reach its first content operation while its
access is still in flight; a read waits a bounded moment for the one it was sent before
deciding it has none. A task that outlives its access asks control to renew it, which
control does only while the task is still this worker's to run.

The material never leaves here. It opens a backend client and is held only for as long
as the grant it came with; nothing writes it down, reports it, or renders it.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import boto3
from botocore.client import Config

from shared.content import (
    BACKEND_FILESYSTEM,
    BACKEND_S3,
    ContentStoreAccess,
    ContentStoreError,
    FabricObjectStore,
    ObjectStoreConfig,
    ScopedContentCredential,
    SharedFilesystemObjectStore,
)
from shared.content.s3_store import S3Client, S3ObjectStore

_AccessKey = tuple[str, str]

# How long before an access ends this worker asks to renew it, so a long upload moves
# onto renewed access without a call ever starting on a session about to lapse.
_RENEW_AHEAD_SEC = 120.0


class ContentAccessDenied(ContentStoreError):
    """This task holds no live access to the shared store in this scope."""


class ContentBackendUnsupported(ContentStoreError):
    """This worker is configured for a store backend it cannot open."""


class ContentAccessRegistry:
    """The stores this worker may open, one per task and scope control granted."""

    def __init__(
        self,
        cfg: ObjectStoreConfig,
        logger: logging.Logger | None = None,
        *,
        request_access: Callable[[str], None] | None = None,
        arrival_wait_sec: float = 5.0,
        renewal_wait_sec: float = 15.0,
    ) -> None:
        self._cfg = cfg
        self._logger = logger or logging.getLogger("content-access")
        self._request_access = request_access
        self._arrival_wait_sec = arrival_wait_sec
        self._renewal_wait_sec = renewal_wait_sec
        self._arrived = threading.Condition()
        self._granted: dict[_AccessKey, ContentStoreAccess] = {}
        self._stores: dict[_AccessKey, FabricObjectStore] = {}
        self._clients: dict[_AccessKey, tuple[str, S3Client]] = {}
        self._opening = threading.Lock()
        self._session: boto3.Session | None = None
        self._renewing: set[str] = set()

    def accept(self, access: ContentStoreAccess) -> None:
        """Take one access control minted for a task of this worker."""
        key = (access.grant.task_id, access.grant.authorization_scope)
        with self._arrived:
            if (superseded := self._granted.get(key)) is not None:
                self._renewing.discard(superseded.grant.grant_id)
            self._granted[key] = access
            self._expire()
            self._arrived.notify_all()

    def store_for(self, task_id: str, scope: str) -> FabricObjectStore:
        """The store this task opens for a scope, or a refusal if it holds none.

        The store resolves the task's access afresh for every call it makes, so a write
        that outlives one access continues on the access renewed under it.
        """
        key = (task_id, scope)
        self._access(key)
        with self._arrived:
            if (store := self._stores.get(key)) is None:
                store = self._open(key)
                self._stores[key] = store
            return store

    def _access(self, key: _AccessKey) -> ContentStoreAccess:
        """This task's live access, or a refusal if it holds none.

        Access that has run out is renewed on request, and a renewal control refuses or
        never relays ends in the same refusal as having none, after a bounded wait.
        Access close to its end is renewed ahead of time, without waiting for it.
        """
        task_id, scope = key
        with self._arrived:
            # An access that has expired is not one still in flight, so only a task
            # that never had one waits out the relay.
            access = (
                self._live(key)
                if key in self._granted
                else self._await_live(key, self._arrival_wait_sec)
            )
        if access is None and self._request_access is not None:
            try:
                self._request_access(task_id)
            except Exception:
                self._logger.warning(
                    "could not ask to renew content store access for %s",
                    task_id,
                    exc_info=True,
                )
            else:
                with self._arrived:
                    access = self._await_live(key, self._renewal_wait_sec)
        if access is None:
            raise ContentAccessDenied(
                f"task {task_id} holds no live content store access in scope {scope}"
            )
        self._renew_ahead(access)
        return access

    def _renew_ahead(self, access: ContentStoreAccess) -> None:
        grant = access.grant
        if self._request_access is None or not grant.expired(
            time.time() + _RENEW_AHEAD_SEC
        ):
            return
        with self._arrived:
            if grant.grant_id in self._renewing:
                return
            self._renewing.add(grant.grant_id)
        try:
            self._request_access(grant.task_id)
        except Exception:
            with self._arrived:
                self._renewing.discard(grant.grant_id)
            self._logger.warning(
                "could not ask to renew content store access for %s",
                grant.task_id,
                exc_info=True,
            )

    def _await_live(self, key: _AccessKey, timeout: float) -> ContentStoreAccess | None:
        """This task's unexpired access, waiting up to ``timeout`` for one to land."""
        self._arrived.wait_for(lambda: self._live(key) is not None, timeout=timeout)
        return self._live(key)

    def _live(self, key: _AccessKey) -> ContentStoreAccess | None:
        access = self._granted.get(key)
        if access is None or not access.grant.expired():
            return access
        self._forget(key)
        return None

    def _open(self, key: _AccessKey) -> FabricObjectStore:
        match self._cfg.backend:
            case x if x == BACKEND_S3:
                return S3ObjectStore(
                    _RenewingClient(lambda: self._client(key)),
                    self._cfg.bucket,
                    prefix=self._cfg.prefix,
                    part_bytes=self._cfg.part_bytes,
                    upload_concurrency=self._cfg.upload_concurrency,
                )
            case x if x == BACKEND_FILESYSTEM:
                return SharedFilesystemObjectStore(self._cfg.filesystem_root)
            case _:
                raise ContentBackendUnsupported(
                    f"unknown content store backend {self._cfg.backend}"
                )

    def _client(self, key: _AccessKey) -> S3Client:
        """The backend client this task's live access opens, opened once per access.

        An upload's part threads all ask at once when its access is renewed, so one of
        them opens the client and the rest take it.
        """
        access = self._access(key)
        with self._opening:
            with self._arrived:
                # Access renewed while this call waited to open is the one to open.
                access = self._live(key) or access
                cached = self._clients.get(key)
            grant_id = access.grant.grant_id
            if cached is not None and cached[0] == grant_id:
                return cached[1]
            client = self._open_s3(access.credential)
            with self._arrived:
                self._clients[key] = (grant_id, client)
            return client

    def _open_s3(self, credential: ScopedContentCredential) -> S3Client:
        material = credential.material
        # Clients open one at a time from this registry's own session: boto3's default
        # session is shared with the rest of the process, and no session is safe to
        # open clients from on several threads at once.
        if self._session is None:
            self._session = boto3.Session()
        client: S3Client = self._session.client(
            "s3",
            endpoint_url=self._cfg.endpoint_url or None,
            aws_access_key_id=material.get("access_key") or None,
            aws_secret_access_key=material.get("secret_key") or None,
            aws_session_token=material.get("session_token") or None,
            region_name=self._cfg.region,
            config=Config(signature_version="s3v4"),
        )
        return client

    def clear(self) -> None:
        """Drop every access, as the incarnation it was granted to has ended."""
        with self._arrived:
            self._granted.clear()
            self._stores.clear()
            self._clients.clear()
            self._renewing.clear()

    def _forget(self, key: _AccessKey) -> None:
        if (access := self._granted.pop(key, None)) is not None:
            self._renewing.discard(access.grant.grant_id)
        self._stores.pop(key, None)
        self._clients.pop(key, None)

    def _expire(self) -> None:
        now = time.time()
        for key, access in list(self._granted.items()):
            if access.grant.expired(now):
                self._forget(key)


class _RenewingClient:
    """An S3 client that is whatever the task's live access opens at each call.

    An abort is housekeeping for an upload this worker started, so it goes out on the
    last client that worked when the task holds no live access any more.
    """

    def __init__(self, resolve: Callable[[], S3Client]) -> None:
        self._resolve = resolve
        self._last: S3Client | None = None

    def _current(self) -> S3Client:
        self._last = self._resolve()
        return self._last

    def put_object(self, **kwargs: Any) -> Any:
        return self._current().put_object(**kwargs)

    def get_object(self, **kwargs: Any) -> Any:
        return self._current().get_object(**kwargs)

    def head_object(self, **kwargs: Any) -> Any:
        return self._current().head_object(**kwargs)

    def create_multipart_upload(self, **kwargs: Any) -> Any:
        return self._current().create_multipart_upload(**kwargs)

    def upload_part(self, **kwargs: Any) -> Any:
        return self._current().upload_part(**kwargs)

    def complete_multipart_upload(self, **kwargs: Any) -> Any:
        return self._current().complete_multipart_upload(**kwargs)

    def abort_multipart_upload(self, **kwargs: Any) -> Any:
        try:
            client = self._current()
        except ContentAccessDenied:
            if self._last is None:
                raise
            client = self._last
        return client.abort_multipart_upload(**kwargs)

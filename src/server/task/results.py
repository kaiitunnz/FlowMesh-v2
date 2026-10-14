"""How the control plane reads a task's result back from the shared content store.

A task's result is the envelope its worker stored before reporting success, named by
the reference that success bound. Reading it hydrates that reference and verifies the
bytes against it, so a result that is missing, corrupt, or not the envelope it names is
refused rather than read as some other value, and a store that cannot be reached is
reported as such, since the result is still there. The store is the only place a result
lives; a reader holds a bounded in-memory cache of recently verified envelopes, which an
immutable reference makes safe to reuse.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from functools import partial

from shared.content import (
    ContentReference,
    ContentStoreError,
    ContentUnavailable,
    DigestAlgorithm,
    FabricObjectStore,
    TreeManifest,
    TreeManifestError,
)
from shared.schemas.result import ResultEnvelope
from shared.schemas.result.binding import (
    NotAResultEnvelope,
    result_envelope,
    skip_envelope,
    skip_envelope_bytes,
)
from shared.tasks.result_binding import ResultBinding

_CACHE_MAX_BYTES = 64 * 1024 * 1024
# The largest envelope the cache keeps, so one large result never displaces the many
# small ones a dispatch loop rereads.
_CACHE_MAX_ENTRY_BYTES = 8 * 1024 * 1024
# What a cached entry holds besides its payload: its reference key, the payload's object
# header and its slot, about 1.3 KiB measured. The budget counts it, so it bounds the
# memory the cache holds.
_CACHE_ENTRY_OVERHEAD_BYTES = 1536
# The members the decoded-tree cache keeps across its trees, about 201 B each, so about
# 50 MiB at the cap. A tree over the cap is kept alone, as the most recent one.
_TREE_MEMBERS_MAX = 1 << 18

type ObjectIdentity = tuple[str, str, str]


class ResultUnreadable(RuntimeError):
    """A bound result is missing from the store, corrupt, or not a result envelope."""


class ResultUnavailable(RuntimeError):
    """The store holding a bound result could not be reached; a retry may succeed."""


class ResultReader:
    """Reads stored result envelopes, verified against the reference naming them."""

    def __init__(
        self, store: FabricObjectStore, *, cache_max_bytes: int = _CACHE_MAX_BYTES
    ) -> None:
        self._store = store
        self._cache_max_bytes = cache_max_bytes
        self._cache: OrderedDict[ContentReference, bytes] = OrderedDict()
        self._cached_bytes = 0
        self._trees: OrderedDict[ObjectIdentity, frozenset[ObjectIdentity]] = (
            OrderedDict()
        )
        self._tree_members = 0
        self._oversize: tuple[ObjectIdentity, frozenset[ObjectIdentity]] | None = None
        self._lock = threading.Lock()

    def read_bytes(self, binding: ResultBinding) -> bytes:
        """The bytes a binding reads as: the stored envelope, or a synthesized skip."""
        if binding.reference is None:
            if binding.skip is None:
                raise ResultUnreadable(f"task {binding.task_id} has no bound result")
            return skip_envelope_bytes(binding)
        return self._hydrate(binding.reference)

    def read(self, binding: ResultBinding) -> ResultEnvelope:
        """The envelope a binding reads as, validated as a result envelope."""
        if binding.reference is None and binding.skip is not None:
            return skip_envelope(binding)
        return _envelope(
            self.read_bytes(binding), f"the stored result of task {binding.task_id}"
        )

    def read_reference(self, reference: ContentReference) -> ResultEnvelope:
        """The envelope a stored result reads as, validated as a result envelope."""
        return _envelope(
            self._hydrate(reference), f"the stored result {reference.content_digest}"
        )

    def read_manifest(self, reference: ContentReference) -> TreeManifest:
        """Return the tree manifest a stored object reads as."""
        try:
            return TreeManifest.decode(self._hydrate(reference))
        except TreeManifestError as exc:
            raise ResultUnreadable(
                f"the stored tree {reference.content_digest} is not a manifest: {exc}"
            ) from exc

    def tree_members(self, reference: ContentReference) -> frozenset[ObjectIdentity]:
        """Return the identities of a stored tree's member objects.

        A tree is decoded once and kept for the most recently read trees, or alone as
        the most recent one too large to share the cache, so a check against its
        members is a set lookup.
        """
        key = reference.identity
        with self._lock:
            if (cached := self._trees.get(key)) is not None:
                self._trees.move_to_end(key)
                return cached
            if self._oversize is not None and self._oversize[0] == key:
                return self._oversize[1]
        scope = reference.authorization_scope
        members = frozenset(
            (scope, DigestAlgorithm.SHA256.value, member.digest)
            for member in self.read_manifest(reference).members
        )
        if len(members) > _TREE_MEMBERS_MAX:
            with self._lock:
                self._oversize = (key, members)
        else:
            with self._lock:
                if key not in self._trees:
                    self._trees[key] = members
                    self._tree_members += len(members)
                    while self._tree_members > _TREE_MEMBERS_MAX:
                        _, evicted = self._trees.popitem(last=False)
                        self._tree_members -= len(evicted)
        return members

    def open(self, reference: ContentReference) -> Iterator[bytes]:
        """Return an object's bytes in chunks, raising before the last one unless they
        verify. Raises ``ResultUnavailable`` while the store cannot be reached and
        ``ResultUnreadable`` for an object that is missing or corrupt."""
        return _classified(partial(self._store.open, reference))

    def open_range(
        self, reference: ContentReference, start: int, length: int
    ) -> Iterator[bytes]:
        """Return ``length`` unverified bytes of an object from ``start``, in chunks."""
        return _classified(partial(self._store.open_range, reference, start, length))

    def verify(self, reference: ContentReference) -> None:
        """Read one object from the store itself, bypassing the cache, and verify it.

        The object streams through and is never held whole. Raises
        ``ResultUnavailable`` while the store cannot be reached and ``ResultUnreadable``
        for an object that is missing or corrupt.
        """
        try:
            for _ in self._store.open(reference):
                pass
        except ContentUnavailable as exc:
            raise ResultUnavailable(str(exc)) from exc
        except ContentStoreError as exc:
            raise ResultUnreadable(str(exc)) from exc

    def verify_stored(self, reference: ContentReference) -> None:
        """Check that one object is in the store at its size, reading none of it.

        Raises ``ResultUnavailable`` while the store cannot be reached and
        ``ResultUnreadable`` for an object that is missing or of another size.
        """
        try:
            size = self._store.stored_size(reference)
        except ContentUnavailable as exc:
            raise ResultUnavailable(str(exc)) from exc
        except ContentStoreError as exc:
            raise ResultUnreadable(str(exc)) from exc
        if size != reference.size_bytes:
            raise ResultUnreadable(
                f"content {reference.content_digest} is stored at {size} bytes, not "
                f"{reference.size_bytes}"
            )

    def _hydrate(self, reference: ContentReference) -> bytes:
        key = reference
        with self._lock:
            if (cached := self._cache.get(key)) is not None:
                self._cache.move_to_end(key)
                return cached
        try:
            data = self._store.hydrate(reference)
        except ContentUnavailable as exc:
            raise ResultUnavailable(str(exc)) from exc
        except ContentStoreError as exc:
            raise ResultUnreadable(str(exc)) from exc
        self._remember(key, data)
        return data

    def _remember(self, key: ContentReference, data: bytes) -> None:
        if len(data) > _CACHE_MAX_ENTRY_BYTES or _charge(data) > self._cache_max_bytes:
            return
        with self._lock:
            if key in self._cache:
                return
            self._cache[key] = data
            self._cached_bytes += _charge(data)
            while self._cached_bytes > self._cache_max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= _charge(evicted)


def _classified(open_chunks: Callable[[], Iterator[bytes]]) -> Iterator[bytes]:
    try:
        yield from open_chunks()
    except ContentUnavailable as exc:
        raise ResultUnavailable(str(exc)) from exc
    except ContentStoreError as exc:
        raise ResultUnreadable(str(exc)) from exc


def _charge(data: bytes) -> int:
    return len(data) + _CACHE_ENTRY_OVERHEAD_BYTES


def _envelope(data: bytes, source: str) -> ResultEnvelope:
    try:
        return result_envelope(data, source)
    except NotAResultEnvelope as exc:
        raise ResultUnreadable(str(exc)) from exc


__all__ = ["ObjectIdentity", "ResultReader", "ResultUnavailable", "ResultUnreadable"]

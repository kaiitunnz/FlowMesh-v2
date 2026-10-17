"""Protecting the members a bound tree manifest names.

A tree binds its manifest; its members are objects of their own that a binding never
names, so they are protected by reading the manifest from the store. The read streams,
holding one window and one member at a time, and a manifest's digest verifies only
once its stream ends, so its members are first held under the expansion's own nonce
and only those held are then protected, by a second read; a corrupt read can hold
digests for a while but never protect one. Until a manifest's members are protected it
stays in its scope's unexpanded set, which keeps the scope out of collection.

A member being retired stops a staged tree's expansion until its retirement ends. A
tree bound with its bytes already stored can have no member being retired, since a
scope is certified before anything in it is retired; finding one un-certifies the
scope.
"""

import logging
from collections.abc import Callable, Iterator, Set
from dataclasses import dataclass
from enum import StrEnum
from itertools import batched
from typing import Any

from shared.content import ContentReference
from shared.content.tree import (
    TREE_MEDIA_TYPE,
    TreeManifestError,
    TreeMember,
    iter_members,
)

from ..clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    RECLAIM_SCOPES_KEY,
    REDIS_CONN_ERRORS,
    RedisClient,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)
from ..task.results import ResultUnavailable, ResultUnreadable
from .pins import CHUNK

# KEYS: protected, pins, scopes, root epoch. ARGV: nonce, scope, epoch, d1..dn.
# Returns -1 when a later root holds the epoch, else how many members are neither
# promoted nor already protected.
_PROMOTE = """
for i = 1, 3 do
  local kind = redis.call('TYPE', KEYS[i]).ok
  if kind ~= 'none' and kind ~= 'set' and kind ~= 'hash' then
    return redis.error_reply('WRONGTYPE ' .. KEYS[i])
  end
end
if redis.call('GET', KEYS[4]) ~= ARGV[3] then
  return -1
end
redis.call('SADD', KEYS[3], ARGV[2])
local missing = 0
for i = 4, #ARGV do
  local current = redis.call('HGET', KEYS[2], ARGV[i])
  local held = false
  if current then
    local kept = {}
    for _, holder in ipairs(cjson.decode(current)) do
      if holder.n == ARGV[1] then
        held = true
      else
        table.insert(kept, holder)
      end
    end
    if held then
      redis.call('SADD', KEYS[1], ARGV[i])
      if #kept == 0 then
        redis.call('HDEL', KEYS[2], ARGV[i])
      else
        redis.call('HSET', KEYS[2], ARGV[i], cjson.encode(kept))
      end
    end
  end
  if not held and redis.call('SISMEMBER', KEYS[1], ARGV[i]) == 0 then
    missing = missing + 1
  end
end
return missing
"""

# KEYS: unexpanded, uncertified, root epoch. ARGV: epoch, manifest, broken-tree marker.
_EXPANDED = """
if redis.call('GET', KEYS[3]) ~= ARGV[1] then
  return 0
end
redis.call('SREM', KEYS[1], ARGV[2])
redis.call('SREM', KEYS[2], ARGV[3])
return 1
"""

# KEYS: protected, pins, unexpanded. ARGV: manifest.
_SETTLE_UNBOUND = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1
    or redis.call('HEXISTS', KEYS[2], ARGV[1]) == 1 then
  return 0
end
redis.call('SREM', KEYS[3], ARGV[1])
return 1
"""

type OpenStream = Callable[[ContentReference], Iterator[bytes]]
type HoldPage = Callable[[str, list[TreeMember]], "TreeOutcome | None"]


class TreeStatus(StrEnum):
    DONE = "done"
    MALFORMED = "malformed"
    """The stored manifest is not a well-formed tree, or not the bytes it names."""
    UNAVAILABLE = "unavailable"
    """The store or the control store could not be read now."""
    REFUSED = "refused"
    """A member's hold was refused."""
    RETIRING = "retiring"
    """A member is being retired; the expansion waits for its retirement to end."""
    INCOMPLETE = "incomplete"
    """A member's hold ended before it was protected; the manifest stays unexpanded
    for an expansion under a fresh hold."""
    FENCED = "fenced"
    """A later root holds the root epoch; this root changes nothing more."""


@dataclass(frozen=True)
class TreeOutcome:
    status: TreeStatus
    reason: str | None = None
    refusal: Any = None
    retiring: tuple[str, ...] = ()
    """The members being retired."""
    retired: tuple[ContentReference, ...] = ()
    """Members retired before, whose current backing the upload's durability check
    verifies."""


_UNREACHABLE: tuple[type[BaseException], ...] = (ResultUnavailable, *REDIS_CONN_ERRORS)
_FENCED = TreeOutcome(TreeStatus.FENCED, "a later root holds the root epoch")


def _broken_tree_marker(digest: str) -> str:
    return f"tree:{digest}"


class TreeExpander:
    """Holds, protects and releases the members of stored tree manifests."""

    def __init__(
        self,
        rds: RedisClient,
        open_stream: OpenStream,
        stored_size: Callable[[ContentReference], int | None],
        logger: logging.Logger | None = None,
    ) -> None:
        self._rds = rds
        self._open = open_stream
        self._stored_size = stored_size
        self._logger = logger or logging.getLogger("tree-expander")

    def hold(self, manifest: ContentReference, hold_page: HoldPage) -> TreeOutcome:
        """Hold every member ``manifest`` names, a page at a time, through
        ``hold_page``; on any failure what it held stays held for the caller to
        release."""
        try:
            for page in self._pages(manifest):
                if (
                    refused := hold_page(manifest.authorization_scope, page)
                ) is not None:
                    return refused
        except TreeManifestError as exc:
            return TreeOutcome(TreeStatus.MALFORMED, str(exc))
        except ResultUnreadable as exc:
            return TreeOutcome(TreeStatus.MALFORMED, str(exc))
        except _UNREACHABLE as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        return TreeOutcome(TreeStatus.DONE)

    def promote(self, manifest: ContentReference, nonce: str) -> TreeOutcome:
        """Protect each member ``nonce`` holds and end its hold, then mark the
        manifest expanded once every member it names is protected.

        Only the root epoch ``nonce`` was minted under promotes. A promotion cut short
        leaves the manifest unexpanded and the rest of its members held, so promoting
        again finishes it.
        """
        scope = manifest.authorization_scope
        epoch = nonce.partition(":")[0]
        keys = [
            reclaim_protected_key(scope),
            reclaim_pins_key(scope),
            RECLAIM_SCOPES_KEY,
            RECLAIM_ROOT_EPOCH_KEY,
        ]
        missing = 0
        try:
            for page in self._pages(manifest):
                unprotected = int(
                    self._rds.sync.eval(
                        _PROMOTE,
                        len(keys),
                        *keys,
                        nonce,
                        scope,
                        epoch,
                        *(m.digest for m in page),
                    )
                )
                if unprotected < 0:
                    return _FENCED
                missing += unprotected
            if missing:
                return TreeOutcome(
                    TreeStatus.INCOMPLETE,
                    f"{missing} member(s) lost their hold before they were protected",
                )
            if not self._rds.sync.eval(
                _EXPANDED,
                3,
                reclaim_unexpanded_key(scope),
                reclaim_uncertified_key(scope),
                RECLAIM_ROOT_EPOCH_KEY,
                epoch,
                manifest.content_digest,
                _broken_tree_marker(manifest.content_digest),
            ):
                return _FENCED
        except (TreeManifestError, ResultUnreadable) as exc:
            return TreeOutcome(TreeStatus.MALFORMED, str(exc))
        except _UNREACHABLE as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        return TreeOutcome(TreeStatus.DONE)

    def release(
        self, manifest: ContentReference, nonce: str, release: Callable[..., int]
    ) -> TreeOutcome:
        """End ``nonce``'s holds on the members ``manifest`` names, as far as it
        reads."""
        scope = manifest.authorization_scope
        try:
            for page in self._pages(manifest):
                release(scope, [m.digest for m in page], nonce)
        except (TreeManifestError, ResultUnreadable):
            # A manifest that did not read whole held no member past where it broke.
            return TreeOutcome(TreeStatus.DONE)
        except _UNREACHABLE as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        return TreeOutcome(TreeStatus.DONE)

    def uncertify(self, manifest: ContentReference, retiring: tuple[str, ...]) -> None:
        """Keep a bound tree's scope out of collection after a member of it was found
        being retired, until the tree expands."""
        scope = manifest.authorization_scope
        self._rds.sync.sadd(
            reclaim_uncertified_key(scope), _broken_tree_marker(manifest.content_digest)
        )
        self._logger.error(
            "Bound tree %s in scope %s has %d member(s) being retired, which no bound "
            "content may be; its bytes may be lost, and the scope is no longer "
            "collected: %s",
            manifest.content_digest,
            scope,
            len(retiring),
            ", ".join(retiring[:5]),
        )

    def manifest_of(self, scope: str, digest: str) -> ContentReference | None:
        """Return the stored manifest ``digest`` names in ``scope``, sized by the
        store, or None when the store holds none."""
        probe = ContentReference(
            authorization_scope=scope,
            content_digest=digest,
            size_bytes=0,
            media_type=TREE_MEDIA_TYPE,
        )
        size = self._stored_size(probe)
        return None if size is None else probe.model_copy(update={"size_bytes": size})

    def expanded(self, manifests: list[ContentReference]) -> set[str]:
        """Return the digests of the manifests whose members are already protected:
        bound, and no longer waiting to be expanded."""
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            for manifest in manifests:
                scope, digest = manifest.authorization_scope, manifest.content_digest
                pipe.sismember(reclaim_protected_key(scope), digest)
                pipe.sismember(reclaim_unexpanded_key(scope), digest)
            flags = pipe.execute()
        return {
            manifest.content_digest
            for manifest, bound, waiting in zip(
                manifests, flags[::2], flags[1::2], strict=True
            )
            if bound and not waiting
        }

    def unexpanded(self, scope: str) -> list[str]:
        return sorted(self._rds.sync.set_members(reclaim_unexpanded_key(scope)))

    def settle_unbound(self, scope: str, digest: str) -> bool:
        """Drop a manifest from the unexpanded set when nothing binds or holds it, as
        after its publication was refused; returns whether it was dropped."""
        return bool(
            self._rds.sync.eval(
                _SETTLE_UNBOUND,
                3,
                reclaim_protected_key(scope),
                reclaim_pins_key(scope),
                reclaim_unexpanded_key(scope),
                digest,
            )
        )

    def reconcile(
        self,
        scopes: list[str],
        expand: Callable[[ContentReference], TreeOutcome],
        stop: Callable[[], bool],
        owed: Set[str] = frozenset(),
    ) -> int:
        """Expand every bound manifest still waiting in ``scopes`` and drop each that
        nothing binds or holds; returns how many still wait, counting one for a pass
        cut short.

        A tree in ``owed`` is a staged tree whose upload is still owed, which its
        publication expands; it is left waiting. The pass ends early when a later root
        holds the epoch or the control store cannot be reached; what it did not reach
        stays unexpanded for the next pass.
        """
        waiting = 0
        try:
            for scope in scopes:
                for digest in self.unexpanded(scope):
                    if stop():
                        return waiting + 1
                    if digest in owed:
                        waiting += 1
                        continue
                    if self.settle_unbound(scope, digest):
                        continue
                    if not self._rds.sync.sismember(
                        reclaim_protected_key(scope), digest
                    ):
                        # Held by a publication still in flight, which expands it.
                        waiting += 1
                        continue
                    outcome = self._reconcile_one(scope, digest, expand)
                    if outcome.status is TreeStatus.FENCED:
                        return waiting + 1
                    if outcome.status is not TreeStatus.DONE:
                        waiting += 1
                        self._logger.warning(
                            "Tree %s in scope %s stays unexpanded: %s",
                            digest,
                            scope,
                            outcome.reason,
                        )
        except REDIS_CONN_ERRORS as exc:
            self._logger.warning("Expanding bound trees waits for Redis: %s", exc)
            return waiting + 1
        return waiting

    def _reconcile_one(
        self,
        scope: str,
        digest: str,
        expand: Callable[[ContentReference], TreeOutcome],
    ) -> TreeOutcome:
        try:
            manifest = self.manifest_of(scope, digest)
        except (ResultUnavailable, OSError) as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, f"cannot be read now: {exc}")
        if manifest is None:
            return TreeOutcome(TreeStatus.MALFORMED, "the store holds no manifest")
        return expand(manifest)

    def scopes(self) -> list[str]:
        return sorted(self._rds.sync.set_members(RECLAIM_SCOPES_KEY))

    def _pages(self, manifest: ContentReference) -> Iterator[list[TreeMember]]:
        for page in batched(iter_members(self._open(manifest)), CHUNK):
            yield list(page)


__all__ = ["TreeExpander", "TreeOutcome", "TreeStatus"]

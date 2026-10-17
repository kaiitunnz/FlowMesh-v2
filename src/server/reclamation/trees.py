"""Protecting the members a bound tree manifest names.

A tree binds its manifest; its members are objects of their own that a binding never
names, so they are protected by reading the manifest from the store. The read streams,
holding one window and one member at a time, and a manifest's digest verifies only
once its stream ends, so its members are first held under the expansion's own nonce
and only those held are then protected, by a second read; a corrupt read can hold
digests for a while but never protect one. Until a manifest's members are protected it
stays in its scope's unexpanded set, which keeps the scope out of collection.
"""

import logging
from collections.abc import Callable, Iterator
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
    RECLAIM_SCOPES_KEY,
    RedisClient,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
)
from ..task.results import ResultUnavailable, ResultUnreadable
from .pins import CHUNK

# KEYS: protected, pins, scopes. ARGV: nonce, scope, d1..dn.
_PROMOTE = """
for _, key in ipairs(KEYS) do
  local kind = redis.call('TYPE', key).ok
  if kind ~= 'none' and kind ~= 'set' and kind ~= 'hash' then
    return redis.error_reply('WRONGTYPE ' .. key)
  end
end
redis.call('SADD', KEYS[3], ARGV[2])
local promoted = 0
for i = 3, #ARGV do
  local current = redis.call('HGET', KEYS[2], ARGV[i])
  if current then
    local kept, held = {}, false
    for _, holder in ipairs(cjson.decode(current)) do
      if holder.n == ARGV[1] then
        held = true
      else
        table.insert(kept, holder)
      end
    end
    if held then
      redis.call('SADD', KEYS[1], ARGV[i])
      promoted = promoted + 1
      if #kept == 0 then
        redis.call('HDEL', KEYS[2], ARGV[i])
      else
        redis.call('HSET', KEYS[2], ARGV[i], cjson.encode(kept))
      end
    end
  end
end
return promoted
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


@dataclass(frozen=True)
class TreeOutcome:
    status: TreeStatus
    reason: str | None = None
    refusal: Any = None


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
        except (ResultUnavailable, OSError) as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        return TreeOutcome(TreeStatus.DONE)

    def promote(self, manifest: ContentReference, nonce: str) -> TreeOutcome:
        """Protect each member ``nonce`` holds and end its hold, then mark the
        manifest expanded."""
        scope = manifest.authorization_scope
        keys = [
            reclaim_protected_key(scope),
            reclaim_pins_key(scope),
            RECLAIM_SCOPES_KEY,
        ]
        try:
            for page in self._pages(manifest):
                self._rds.sync.eval(
                    _PROMOTE, len(keys), *keys, nonce, scope, *(m.digest for m in page)
                )
            self._rds.sync.srem(reclaim_unexpanded_key(scope), manifest.content_digest)
        except (TreeManifestError, ResultUnreadable) as exc:
            return TreeOutcome(TreeStatus.MALFORMED, str(exc))
        except (ResultUnavailable, OSError) as exc:
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
        except (ResultUnavailable, OSError) as exc:
            return TreeOutcome(TreeStatus.UNAVAILABLE, str(exc))
        return TreeOutcome(TreeStatus.DONE)

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
        with self._rds.sync.control_pipeline(transaction=False) as pipe:
            pipe.sismember(reclaim_protected_key(scope), digest)
            pipe.hexists(reclaim_pins_key(scope), digest)
            bound, held = pipe.execute()
        if bound or held:
            return False
        self._rds.sync.srem(reclaim_unexpanded_key(scope), digest)
        return True

    def reconcile(
        self,
        scopes: list[str],
        expand: Callable[[ContentReference], TreeOutcome],
        stop: Callable[[], bool],
    ) -> int:
        """Expand every bound manifest still waiting in ``scopes`` and drop each that
        nothing binds or holds; returns how many still wait."""
        waiting = 0
        for scope in scopes:
            for digest in self.unexpanded(scope):
                if stop():
                    return waiting + 1
                if self.settle_unbound(scope, digest):
                    continue
                if not self._rds.sync.sismember(reclaim_protected_key(scope), digest):
                    # Held by a publication still in flight, which expands it.
                    waiting += 1
                    continue
                try:
                    manifest = self.manifest_of(scope, digest)
                except (ResultUnavailable, OSError) as exc:
                    self._logger.warning("Tree %s cannot be read now: %s", digest, exc)
                    waiting += 1
                    continue
                outcome = (
                    expand(manifest)
                    if manifest is not None
                    else TreeOutcome(
                        TreeStatus.MALFORMED, "the store holds no manifest"
                    )
                )
                if outcome.status is not TreeStatus.DONE:
                    waiting += 1
                    self._logger.warning(
                        "Tree %s in scope %s stays unexpanded: %s",
                        digest,
                        scope,
                        outcome.reason,
                    )
        return waiting

    def scopes(self) -> list[str]:
        return sorted(self._rds.sync.set_members(RECLAIM_SCOPES_KEY))

    def _pages(self, manifest: ContentReference) -> Iterator[list[TreeMember]]:
        for page in batched(iter_members(self._open(manifest)), CHUNK):
            yield list(page)


__all__ = ["TreeExpander", "TreeOutcome", "TreeStatus"]

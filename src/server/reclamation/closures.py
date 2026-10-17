"""The members of bound trees, read once per tree.

A tree binding names only its manifest, so a tree's members are bound through it. Each
manifest is streamed once, verified by size and digest, and its members marked in its
scope's closure, which keeps every tree it has read: a manifest is immutable, so what
it names never changes. A manifest that is not a well-formed tree, or that the store
holds missing or corrupt, is remembered as such and not read again by this root; one
the store cannot serve now is read again at the next sweep.
"""

import logging
from array import array
from collections import defaultdict
from collections.abc import Callable, Generator, Iterator
from enum import StrEnum

from shared.content import ContentReference
from shared.content.tree import TreeManifestError, iter_members

from ..task.results import ResultUnavailable, ResultUnreadable
from .mark import DigestSet, Mark, digest_words
from .pacing import Pacer

# Trees and members read between pauses.
READS_PER_PAUSE = 1000

type OpenContent = Callable[[ContentReference], Iterator[bytes]]


class Expansion(StrEnum):
    """What reading one bound tree's manifest established."""

    EXPANDED = "expanded"
    """Its members are in the closure."""
    MALFORMED = "malformed"
    """It is not a well-formed tree, or the store holds it missing or corrupt, so its
    members are unknown."""
    UNAVAILABLE = "unavailable"
    """The store could not serve it now."""
    STOPPED = "stopped"
    """The pass stopped before the read finished, so its members are unknown."""


class TreeClosures:
    """The members of every tree read so far, by scope."""

    def __init__(self, open_content: OpenContent, logger: logging.Logger) -> None:
        self._open = open_content
        self._logger = logger
        self._mark = Mark()
        self._expanded: defaultdict[str, DigestSet] = defaultdict(DigestSet)
        self._malformed: defaultdict[str, DigestSet] = defaultdict(DigestSet)

    @property
    def mark(self) -> Mark:
        return self._mark

    @property
    def nbytes(self) -> int:
        """Bytes the closures hold: their mark and the trees they have settled."""
        return self._mark.nbytes + sum(
            digests.nbytes
            for settled in (self._expanded, self._malformed)
            for digests in settled.values()
        )

    def known(self, tree: ContentReference) -> Expansion | None:
        scope, digest = tree.authorization_scope, tree.content_digest
        if scope in self._expanded and digest in self._expanded[scope]:
            return Expansion.EXPANDED
        if scope in self._malformed and digest in self._malformed[scope]:
            return Expansion.MALFORMED
        return None

    def expand(self, tree: ContentReference, pacer: Pacer) -> Expansion:
        """Read the tree's manifest unless an earlier read settled it, ticking
        ``pacer`` per member read."""
        if (settled := self.known(tree)) is not None:
            return settled
        scope = tree.authorization_scope
        # Two words per member, folded into the closure only once the bytes verify.
        words = array("Q")
        chunks = self._open(tree)
        try:
            for member in iter_members(chunks):
                words.extend(digest_words(member.digest))
                if pacer.tick():
                    return Expansion.STOPPED
        except TreeManifestError as exc:
            self._malformed[scope].add(tree.content_digest)
            self._logger.error(
                "Content audit cannot read bound tree %s in scope %s: %s",
                tree.content_digest,
                scope,
                exc,
            )
            return Expansion.MALFORMED
        except ResultUnreadable as exc:
            self._malformed[scope].add(tree.content_digest)
            self._logger.error(
                "Content audit finds bound tree %s in scope %s unreadable: %s",
                tree.content_digest,
                scope,
                exc,
            )
            return Expansion.MALFORMED
        except ResultUnavailable:
            return Expansion.UNAVAILABLE
        finally:
            if isinstance(chunks, Generator):
                chunks.close()
        self._mark.add_all(
            scope, zip(words[0::2], words[1::2], strict=True), len(words) // 2
        )
        self._expanded[scope].add(tree.content_digest)
        return Expansion.EXPANDED

    def compact(self) -> None:
        """Pack the trees settled since the last compaction."""
        for settled in (self._expanded, self._malformed):
            for digests in settled.values():
                digests.compact()


__all__ = ["READS_PER_PAUSE", "Expansion", "OpenContent", "TreeClosures"]

"""The members of bound trees, read once per tree.

A tree binding names only its manifest, so a tree's members are bound through it. Each
manifest is streamed once, verified by size and digest, and its members marked in its
scope's closure, which keeps every tree it has read: a manifest is immutable, so what
it names never changes. A manifest that is not a well-formed tree, or that the store
holds missing or corrupt, is remembered as such and not read again by this root; one
the store cannot serve now is read again at the next sweep.
"""

import logging
import threading
from array import array
from collections.abc import Callable, Generator, Iterator
from enum import StrEnum

from shared.content import ContentReference
from shared.content.tree import TreeManifestError, iter_members

from ..task.results import ResultUnavailable, ResultUnreadable
from .mark import Mark, digest_words
from .pacing import DUTY, MIN_PAUSE_SEC, Pacer

# Members read between pauses.
MEMBERS_PER_PAUSE = 1000

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


class TreeClosures:
    """The members of every tree read so far, by scope."""

    def __init__(
        self,
        open_content: OpenContent,
        logger: logging.Logger,
        *,
        members_per_pause: int = MEMBERS_PER_PAUSE,
        duty: float = DUTY,
        min_pause_sec: float = MIN_PAUSE_SEC,
    ) -> None:
        self._open = open_content
        self._logger = logger
        self._members_per_pause = members_per_pause
        self._duty = duty
        self._min_pause = min_pause_sec
        self._mark = Mark()
        self._expanded: set[tuple[str, str]] = set()
        self._malformed: dict[tuple[str, str], str] = {}
        self.reads = 0

    @property
    def mark(self) -> Mark:
        return self._mark

    def known(self, tree: ContentReference) -> Expansion | None:
        identity = (tree.authorization_scope, tree.content_digest)
        if identity in self._expanded:
            return Expansion.EXPANDED
        if identity in self._malformed:
            return Expansion.MALFORMED
        return None

    def expand(self, tree: ContentReference, stop: threading.Event) -> Expansion:
        """Read the tree's manifest unless an earlier read settled it."""
        if (settled := self.known(tree)) is not None:
            return settled
        identity = (tree.authorization_scope, tree.content_digest)
        self.reads += 1
        # Two words per member, folded into the closure only once the bytes verify.
        words = array("Q")
        pacer = Pacer(
            stop,
            self._members_per_pause,
            duty=self._duty,
            min_pause_sec=self._min_pause,
        )
        chunks = self._open(tree)
        try:
            for member in iter_members(chunks):
                words.extend(digest_words(member.digest))
                if pacer.tick():
                    return Expansion.STOPPED
        except TreeManifestError as exc:
            self._malformed[identity] = str(exc)
            self._logger.error(
                "Content audit cannot read bound tree %s in scope %s: %s",
                tree.content_digest,
                tree.authorization_scope,
                exc,
            )
            return Expansion.MALFORMED
        except ResultUnreadable as exc:
            self._malformed[identity] = str(exc)
            self._logger.error(
                "Content audit finds bound tree %s in scope %s unreadable: %s",
                tree.content_digest,
                tree.authorization_scope,
                exc,
            )
            return Expansion.MALFORMED
        except ResultUnavailable:
            return Expansion.UNAVAILABLE
        finally:
            if isinstance(chunks, Generator):
                chunks.close()
        self._mark.add_all(
            tree.authorization_scope,
            zip(words[0::2], words[1::2], strict=True),
            len(words) // 2,
        )
        self._expanded.add(identity)
        return Expansion.EXPANDED

    def malformed(self, scope: str) -> dict[str, str]:
        """The trees of ``scope`` known not to be well formed, with why."""
        return {
            digest: why
            for (tree_scope, digest), why in self._malformed.items()
            if tree_scope == scope
        }


__all__ = ["Expansion", "OpenContent", "TreeClosures"]

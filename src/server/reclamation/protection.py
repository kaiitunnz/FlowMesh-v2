"""The protection index: every content identity some binding names, per scope.

An identity enters the index in the same transaction that commits the binding naming
it and never leaves it, since every bound object is retained. The index is derived
evidence the collector reads; it is not a binding of its own.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from shared.content.reference import ContentReference

from ..clients.redis import RECLAIM_SCOPES_KEY, reclaim_protected_key


def by_scope(references: Iterable[ContentReference]) -> dict[str, set[str]]:
    """The digests of ``references``, grouped by the scope isolating them."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for reference in references:
        grouped[reference.authorization_scope].add(reference.content_digest)
    return grouped


def queue_protection(pipe: Any, references: Iterable[ContentReference]) -> None:
    """Queue the protection of every identity ``references`` name on ``pipe``."""
    grouped = by_scope(references)
    if not grouped:
        return
    pipe.sadd(RECLAIM_SCOPES_KEY, *grouped)
    for scope, digests in grouped.items():
        pipe.sadd(reclaim_protected_key(scope), *sorted(digests))


__all__ = ["by_scope", "queue_protection"]

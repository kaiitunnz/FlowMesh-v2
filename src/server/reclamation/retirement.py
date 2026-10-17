"""Whether a staged upload's objects may settle durable, given their retirement.

A staged object is accepted before it is stored, so its worker's conditional write may
have found an object of the same digest that was being retired and taken it as its
own. Before such an upload settles durable, each object it names is checked: one still
being retired waits for the retirement to end, and one retired before is durable only
while the store holds a current version of it; otherwise its worker stores it again
from what it kept.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from shared.content import ContentReference

from .pins import Retirement

type Backed = Callable[[ContentReference], bool | None]


class RetirementIndex(Protocol):
    def retirement(
        self, scope: str, digests: Sequence[str]
    ) -> dict[str, Retirement]: ...


class StagedStanding(StrEnum):
    CLEAR = "clear"
    RETIRING = "retiring"
    """An object is being retired; the upload waits, storing nothing, and is reported
    again."""
    MISSING = "missing"
    """An object retired before has no current version; its worker stores it again."""
    UNKNOWN = "unknown"
    """The store could not say whether an object retired before is there."""


@dataclass(frozen=True)
class StagedVerdict:
    standing: StagedStanding
    digests: tuple[str, ...] = ()


class StagedRetirement:
    """Checks a staged upload's objects against their retirement."""

    def __init__(self, index: RetirementIndex, backed: Backed) -> None:
        self._index = index
        self._backed = backed

    def check(self, references: Iterable[ContentReference]) -> StagedVerdict:
        """Return whether the objects ``references`` name may settle durable."""
        retiring: list[str] = []
        retired: list[ContentReference] = []
        for scope, by_digest in _by_scope(references).items():
            standing = self._index.retirement(scope, list(by_digest))
            for digest, state in standing.items():
                if state is Retirement.RETIRING:
                    retiring.append(digest)
                elif state is Retirement.RETIRED:
                    retired.append(by_digest[digest])
        if retiring:
            return StagedVerdict(StagedStanding.RETIRING, tuple(sorted(retiring)))
        missing: list[str] = []
        unknown = False
        for reference in retired:
            match self._backed(reference):
                case False:
                    missing.append(reference.content_digest)
                case None:
                    unknown = True
        if missing:
            return StagedVerdict(StagedStanding.MISSING, tuple(sorted(missing)))
        if unknown:
            return StagedVerdict(StagedStanding.UNKNOWN)
        return StagedVerdict(StagedStanding.CLEAR)

    def retiring(self, references: Iterable[ContentReference]) -> list[str]:
        """Return the digests of ``references`` being retired now."""
        return sorted(
            digest
            for scope, by_digest in _by_scope(references).items()
            for digest, state in self._index.retirement(scope, list(by_digest)).items()
            if state is Retirement.RETIRING
        )


def _by_scope(
    references: Iterable[ContentReference],
) -> dict[str, dict[str, ContentReference]]:
    grouped: dict[str, dict[str, ContentReference]] = defaultdict(dict)
    for reference in references:
        grouped[reference.authorization_scope][reference.content_digest] = reference
    return grouped


__all__ = [
    "Backed",
    "RetirementIndex",
    "StagedRetirement",
    "StagedStanding",
    "StagedVerdict",
]

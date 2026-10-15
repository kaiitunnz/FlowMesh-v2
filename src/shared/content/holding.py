"""What a worker tells control about the copies its cache holds.

A worker reports its holding as it changes: the copies it kept and the copies it dropped
since its last report, and its whole holding when it attaches under a new registration
or control asks for it. Reports of one worker registration are numbered from one, so
control can tell when one went missing and ask for the whole holding again.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The control frame asking a worker to report its whole holding.
HOLDING_RESYNC_FRAME = "content_holding_resync"

type HeldCopy = tuple[str, str]
"""One copy, as the scope and digest naming its object."""


@dataclass(frozen=True)
class HoldingReport:
    """One report of a worker's holding.

    A ``whole`` report names every copy in ``added`` and replaces what control recorded
    for the worker; any other names what changed since the report numbered one less.
    """

    seq: int
    added: tuple[HeldCopy, ...] = ()
    dropped: tuple[HeldCopy, ...] = ()
    whole: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "added": [[scope, digest] for scope, digest in self.added],
            "dropped": [[scope, digest] for scope, digest in self.dropped],
            "whole": self.whole,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HoldingReport":
        return cls(
            seq=int(payload["seq"]),
            added=_copies(payload.get("added") or ()),
            dropped=_copies(payload.get("dropped") or ()),
            whole=bool(payload.get("whole")),
        )


def _copies(raw: Sequence[Sequence[Any]]) -> tuple[HeldCopy, ...]:
    return tuple((str(scope), str(digest)) for scope, digest in raw)

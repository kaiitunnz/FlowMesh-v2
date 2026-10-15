"""What a worker tells control about the copies its cache holds.

A worker reports its holding as it changes: the copies it kept and the copies it dropped
since its last report, and its whole holding when it attaches under a new registration
or control asks for it. Reports of one worker registration are numbered from one, its
whole holding first, so control can tell when one went missing and ask for the whole
holding again.
"""

from pydantic import BaseModel, ConfigDict

# The control frame asking a worker to report its whole holding.
HOLDING_RESYNC_FRAME = "content_holding_resync"

type HeldCopy = tuple[str, str]
"""One copy, as the scope and digest naming its object."""


class HoldingReport(BaseModel):
    """One report of a worker's holding.

    A ``whole`` report names every copy in ``added`` and replaces what control recorded
    for the worker; any other names what changed since the report numbered one less.
    """

    model_config = ConfigDict(frozen=True)

    seq: int
    added: tuple[HeldCopy, ...] = ()
    dropped: tuple[HeldCopy, ...] = ()
    whole: bool = False

"""Opaque cursors over a list ordered by a stable identity.

A cursor encodes the identity of the entry it names, so it keeps naming the same
position while entries are added or removed around it.
"""

import base64
import binascii
import json
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Any


class InvalidCursor(ValueError):
    """A cursor that is not one this server issued."""


def encode_cursor(identity: Sequence[Any]) -> str:
    raw = json.dumps(list(identity), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: str) -> list[Any]:
    """The identity a cursor encodes; raises InvalidCursor for anything else."""
    try:
        identity = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
    except (binascii.Error, UnicodeError, ValueError) as exc:
        raise InvalidCursor(f"invalid cursor {cursor!r}") from exc
    if not isinstance(identity, list):
        raise InvalidCursor(f"invalid cursor {cursor!r}")
    return identity


def page_slice(
    keys: Sequence[Any],
    limit: int,
    after: Any = None,
    before: Any = None,
    newest: bool = False,
) -> slice:
    """The ``limit`` positions of ascending ``keys`` strictly after, or strictly
    before, a bound; with neither, the first ``limit``, or the last when ``newest``."""
    if after is not None:
        start = bisect_right(keys, after)
        return slice(start, start + limit)
    if before is not None:
        end = bisect_left(keys, before)
        return slice(max(0, end - limit), end)
    if newest:
        return slice(max(0, len(keys) - limit), len(keys))
    return slice(0, limit)


__all__ = ["InvalidCursor", "decode_cursor", "encode_cursor", "page_slice"]

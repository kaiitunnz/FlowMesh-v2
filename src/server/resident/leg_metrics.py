"""Per-leg counters for resident invocation traffic.

A resident invocation's payload crosses two legs: the source-to-root leg an origin
worker carries over its outbound attachment, and the target leg into the selected
replica. They are counted separately, per transport.
"""

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class LegCounter:
    frames: int = 0
    payload_bytes: int = 0


@dataclass
class ResidentLegMetrics:
    """Frame and byte counts per ``(leg, transport)``."""

    _counters: dict[tuple[str, str], LegCounter] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record(self, leg: str, transport: str, payload_bytes: int) -> None:
        with self._lock:
            counter = self._counters.setdefault((leg, transport), LegCounter())
            counter.frames += 1
            counter.payload_bytes += payload_bytes

    def snapshot(self) -> list[dict[str, str | int]]:
        with self._lock:
            return [
                {
                    "leg": leg,
                    "transport": transport,
                    "frames": counter.frames,
                    "payload_bytes": counter.payload_bytes,
                }
                for (leg, transport), counter in sorted(self._counters.items())
            ]


__all__ = ["LegCounter", "ResidentLegMetrics"]

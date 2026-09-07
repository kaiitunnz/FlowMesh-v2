import datetime
import time


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def parse_iso_datetime(value: str | None) -> datetime.datetime | None:
    """Parse ISO 8601 → ``datetime``; ``None`` if empty, raises on malformed."""
    if not value:
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_iso_ts(value: str | None) -> float:
    """ISO 8601 → Unix timestamp; ``time.time()`` on missing / malformed."""
    try:
        dt = parse_iso_datetime(value)
    except ValueError:
        return time.time()
    return dt.timestamp() if dt else time.time()

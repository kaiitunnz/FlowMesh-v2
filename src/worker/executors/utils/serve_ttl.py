import time


def serve_deadline(
    ttl_seconds: float | None,
    elapsed_sec: float | None,
    default_ttl_sec: float,
    max_ttl_sec: float,
) -> float:
    """Return the epoch second a serve task's run ends.

    The TTL counts from the task's first start across re-runs, so a re-run serves only
    what remains of it; ``max_ttl_sec`` caps the whole life.
    """
    ttl = min(ttl_seconds or default_ttl_sec, max_ttl_sec)
    return time.time() + ttl - (elapsed_sec or 0.0)

import time

from shared.utils.parsing import parse_float_env

_DEFAULT_TTL_SEC = 3600.0
_MAX_TTL_SEC = 86400.0


def serve_deadline(ttl_seconds: float | None, elapsed_sec: float | None) -> float:
    """Return the epoch second a serve task's run ends.

    The TTL counts from the task's first start across re-runs, so a re-run serves only
    what remains of it; ``SERVE_MAX_TTL_SEC`` caps the whole life.
    """
    ttl = min(
        ttl_seconds or parse_float_env("SERVE_DEFAULT_TTL_SEC", _DEFAULT_TTL_SEC),
        parse_float_env("SERVE_MAX_TTL_SEC", _MAX_TTL_SEC),
    )
    return time.time() + ttl - (elapsed_sec or 0.0)

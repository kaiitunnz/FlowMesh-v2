"""The event monitor validates the base URL it advertises the gated serve route on.

A well-formed ``http``/``https`` URL passes through; anything else falls back to a safe
default rather than advertising a broken serve route.
"""

import logging

from server.services.monitoring import EventMonitor

_FALLBACK = "http://localhost:8000"


def _validator() -> EventMonitor:
    monitor = EventMonitor.__new__(EventMonitor)
    monitor._logger = logging.getLogger("test-monitor")
    return monitor


def test_valid_http_and_https_urls_pass_through() -> None:
    validate = _validator()._validate_server_base_url
    assert validate("http://host:8000") == "http://host:8000"
    assert validate("https://api.example.com") == "https://api.example.com"


def test_a_malformed_or_non_http_url_falls_back() -> None:
    validate = _validator()._validate_server_base_url
    assert validate("not-a-url") == _FALLBACK
    assert validate("ftp://host:8000") == _FALLBACK
    assert validate("") == _FALLBACK
    # A scheme with no host cannot address the serve route, so it falls back too.
    assert validate("http://") == _FALLBACK

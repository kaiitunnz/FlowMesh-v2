"""The Redis clients log their URLs without the injected password."""

import logging

import pytest

from server.clients.redis import AsyncRedisClient, SyncRedisClient

_URL = "redis://flowmesh:pw-SECRET@127.0.0.1:1/0"


def test_a_failed_connection_logs_no_password(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit):
        SyncRedisClient(_URL, _URL, logging.getLogger("redis-client-logging"))

    assert "Failed to connect" in caplog.text
    assert "pw-SECRET" not in caplog.text


def test_a_rejected_url_logs_no_password(caplog: pytest.LogCaptureFixture):
    url = "unknown://flowmesh:pw-SECRET@127.0.0.1:1/0"
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit):
        AsyncRedisClient(url, url, logging.getLogger("redis-client-logging"))

    assert "Failed to connect" in caplog.text
    assert "pw-SECRET" not in caplog.text

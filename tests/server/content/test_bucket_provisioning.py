"""The control plane makes sure the bucket the fabric's content lives in exists."""

import logging
from typing import Any
from unittest import mock

from server.content import ensure_bucket
from shared.content import ObjectStoreConfig

_CFG = ObjectStoreConfig(
    endpoint_url="http://store:9000",
    bucket="flowmesh-content",
    access_key="key",
    secret_key="secret",
)


def _client(monkeypatch: Any, client: mock.Mock) -> mock.Mock:
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: client)
    return client


def test_an_existing_bucket_is_left_alone(monkeypatch: Any) -> None:
    client = _client(monkeypatch, mock.Mock())
    ensure_bucket(_CFG, logging.getLogger("t"))
    client.head_bucket.assert_called_once_with(Bucket="flowmesh-content")
    client.create_bucket.assert_not_called()


def test_a_missing_bucket_is_created(monkeypatch: Any) -> None:
    client = mock.Mock()
    client.head_bucket.side_effect = RuntimeError("404")
    _client(monkeypatch, client)
    ensure_bucket(_CFG, logging.getLogger("t"))
    client.create_bucket.assert_called_once_with(Bucket="flowmesh-content")


def test_a_store_that_refuses_does_not_stop_the_server(monkeypatch: Any) -> None:
    """A store whose credential may not create a bucket is a configuration the
    deployment owns; it surfaces on the first write, not as a failed startup."""
    client = mock.Mock()
    client.head_bucket.side_effect = RuntimeError("404")
    client.create_bucket.side_effect = RuntimeError("AccessDenied")
    _client(monkeypatch, client)
    ensure_bucket(_CFG, logging.getLogger("t"))

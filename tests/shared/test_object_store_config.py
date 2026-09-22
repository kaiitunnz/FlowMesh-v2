"""Where a process finds the shared content store, read from its own environment."""

from pathlib import Path
from typing import Any

from shared.content import BACKEND_S3, ObjectStoreConfig

_VARS = (
    "CONTENT_STORE_BACKEND",
    "CONTENT_STORE_BUCKET",
    "CONTENT_STORE_REGION",
    "CONTENT_STORE_ENDPOINT_URL",
    "CONTENT_STORE_PREFIX",
)


def _clear(monkeypatch: Any) -> None:
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_an_unset_environment_describes_the_default_store(monkeypatch: Any) -> None:
    _clear(monkeypatch)
    cfg = ObjectStoreConfig.from_env(Path("/data"))
    assert cfg.backend == BACKEND_S3
    assert cfg.bucket == "flowmesh-content"
    assert cfg.region == "us-east-1"


def test_an_empty_value_means_unset_rather_than_empty(monkeypatch: Any) -> None:
    """A worker is handed its node's environment, so a variable the node never set
    arrives set-but-empty. Reading that as an empty bucket makes every write fail
    parameter validation instead of reaching the store the deployment configured."""
    _clear(monkeypatch)
    for name in (
        "CONTENT_STORE_BACKEND",
        "CONTENT_STORE_BUCKET",
        "CONTENT_STORE_REGION",
    ):
        monkeypatch.setenv(name, "")

    cfg = ObjectStoreConfig.from_env(Path("/data"))
    assert cfg.backend == BACKEND_S3
    assert cfg.bucket == "flowmesh-content"
    assert cfg.region == "us-east-1"


def test_a_configured_store_is_read_as_given(monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("CONTENT_STORE_BUCKET", "fabric-objects")
    monkeypatch.setenv("CONTENT_STORE_REGION", "ap-southeast-1")
    monkeypatch.setenv("CONTENT_STORE_ENDPOINT_URL", "http://store:9000")

    cfg = ObjectStoreConfig.from_env(Path("/data"))
    assert cfg.bucket == "fabric-objects"
    assert cfg.region == "ap-southeast-1"
    assert cfg.endpoint_url == "http://store:9000"

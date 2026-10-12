"""What a scope's minted session is cut to, and when a new one is cut."""

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from server.content import StsScopedCredentialMinter
from server.content.credentials import scope_policy
from shared.content import ContentOperationKind, ObjectStoreConfig
from shared.content.s3_store import S3ObjectStore

_OPS = (ContentOperationKind.READ, ContentOperationKind.WRITE)
_CFG = ObjectStoreConfig(
    endpoint_url="http://store:9000",
    bucket="flowmesh-content",
    access_key="key",
    secret_key="secret",
)


class _FakeSts:
    def __init__(self, expires_in_sec: float | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._expires_in_sec = expires_in_sec

    def assume_role(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        n = len(self.calls)
        session: dict[str, Any] = {
            "AccessKeyId": f"ak-{n}",
            "SecretAccessKey": f"sk-{n}",
            "SessionToken": f"st-{n}",
        }
        if self._expires_in_sec is not None:
            session["Expiration"] = datetime.now(UTC) + timedelta(
                seconds=self._expires_in_sec
            )
        return {"Credentials": session}


class _UnusedWrites:
    """The multipart calls an S3 client offers, which these reads never make."""

    def head_object(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def create_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def upload_part(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def complete_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")

    def abort_multipart_upload(self, **kwargs: Any) -> Any:
        raise AssertionError("not called here")


def test_the_policy_covers_exactly_the_scopes_own_keys() -> None:
    """The policy's resource has to match where objects are actually written.

    The key layout puts the scope first precisely so a prefix policy can express it; if
    the two ever disagree, a session is cut for a prefix its own writes never reach.
    """
    written: dict[str, Any] = {}

    class _Recording(_UnusedWrites):
        def put_object(self, **kwargs: Any) -> Any:
            written.update(kwargs)

        def get_object(self, **kwargs: Any) -> Any:
            raise AssertionError("not read here")

    S3ObjectStore(_Recording(), _CFG.bucket, prefix=_CFG.prefix).write(
        "tenant-a", b"body"
    )

    policy = scope_policy(_CFG.bucket, _CFG.prefix, "tenant-a", _OPS)
    assert written["Key"].startswith("tenant-a/")
    assert "arn:aws:s3:::flowmesh-content/tenant-a/*" in policy


def test_a_prefixed_deployment_scopes_under_its_own_prefix() -> None:
    policy = scope_policy("bucket", "fabric", "tenant-a", _OPS)

    assert "arn:aws:s3:::bucket/fabric/tenant-a/*" in policy


def test_a_read_only_grant_asks_for_no_write_action() -> None:
    policy = scope_policy("bucket", "", "tenant-a", (ContentOperationKind.READ,))

    assert "s3:GetObject" in policy and "s3:PutObject" not in policy


def test_a_write_grant_may_abort_its_uploads_and_nothing_more() -> None:
    policy = json.loads(scope_policy("bucket", "", "tenant-a", _OPS))
    actions = {a for s in policy["Statement"] for a in _as_list(s["Action"])}

    assert actions == {"s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"}


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def test_one_session_serves_a_scopes_tasks_until_it_nears_its_end() -> None:
    sts = _FakeSts()
    minter = StsScopedCredentialMinter(_CFG, sts)

    first = minter.mint("tenant-a", _OPS, 900.0)
    again = minter.mint("tenant-a", _OPS, 900.0)

    assert first.credential.material == again.credential.material
    assert len(sts.calls) == 1


def test_another_scope_gets_its_own_session() -> None:
    sts = _FakeSts()
    minter = StsScopedCredentialMinter(_CFG, sts)

    minter.mint("tenant-a", _OPS, 900.0)
    minter.mint("tenant-b", _OPS, 900.0)

    assert len(sts.calls) == 2
    assert [c["Policy"].count("tenant-a") for c in sts.calls][0] == 1
    assert "tenant-b" in sts.calls[1]["Policy"]


def test_a_session_near_its_end_is_cut_again() -> None:
    """Reuse stops before the session does, so no task starts on one about to lapse."""
    sts = _FakeSts(expires_in_sec=30.0)
    minter = StsScopedCredentialMinter(_CFG, sts)

    minter.mint("tenant-a", _OPS, 900.0)
    minter.mint("tenant-a", _OPS, 900.0)

    assert len(sts.calls) == 2


def test_a_renewal_inside_a_workers_lead_gets_a_later_expiry() -> None:
    """A session a worker is already asking to replace is never handed back to it."""
    sts = _FakeSts(expires_in_sec=100.0)
    minter = StsScopedCredentialMinter(_CFG, sts)
    first = minter.mint("tenant-a", _OPS, 900.0)

    sts._expires_in_sec = 900.0
    renewed = minter.mint("tenant-a", _OPS, 900.0)

    assert len(sts.calls) == 2
    assert renewed.expires_at_epoch > first.expires_at_epoch + 600.0


def test_the_grant_never_outlives_the_session_it_opens() -> None:
    """A grant that claimed longer than its material would read live while refused."""
    sts = _FakeSts(expires_in_sec=120.0)

    minted = StsScopedCredentialMinter(_CFG, sts).mint("tenant-a", _OPS, 900.0)

    assert minted.expires_at_epoch < time.time() + 900.0

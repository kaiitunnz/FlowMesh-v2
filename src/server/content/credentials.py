"""Cutting backend session material for one authorization scope.

Two realizations, and the difference between them is what isolation the deployment
actually gets. An S3-compatible store issues a short-lived session limited to the
scope's own prefix, so a worker holding it cannot read another scope's content whatever
it asks for. A deployment that hands out its own standing credential instead gives every
scope the same access, which is why it has to be asked for explicitly and says so on the
way up.
"""

import json
import logging
from typing import Any

from shared.content import (
    ContentOperationKind,
    ObjectStoreConfig,
    ScopedContentCredential,
)

_READ_ACTIONS = ("s3:GetObject",)
_WRITE_ACTIONS = ("s3:PutObject",)


def scope_policy(bucket: str, prefix: str, scope: str, operations: tuple) -> str:
    """An S3 policy permitting exactly these operations under one scope's prefix."""
    root = f"{prefix.strip('/')}/{scope}" if prefix.strip("/") else scope
    actions: list[str] = []
    if ContentOperationKind.READ in operations:
        actions.extend(_READ_ACTIONS)
    if ContentOperationKind.WRITE in operations:
        actions.extend(_WRITE_ACTIONS)
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": actions,
                    "Resource": [f"arn:aws:s3:::{bucket}/{root}/*"],
                }
            ],
        }
    )


def ensure_bucket(cfg: ObjectStoreConfig, logger: logging.Logger) -> None:
    """Make sure the bucket the fabric's content lives in exists.

    The control plane holds the deployment credential a scoped session is cut from, so
    it is the one thing that can create the bucket; the sessions it hands out reach one
    scope's prefix and never the bucket itself. Creating it here is what lets a
    deployment bring its own store up beside the fabric and have content land in it with
    nothing else to configure. An existing bucket, or a store whose credential may not
    create one, leaves it alone.
    """
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url or None,
        aws_access_key_id=cfg.access_key or None,
        aws_secret_access_key=cfg.secret_key or None,
        region_name=cfg.region,
        config=Config(signature_version="s3v4"),
    )
    try:
        client.head_bucket(Bucket=cfg.bucket)
        return
    except Exception:  # noqa: BLE001 - absent, unreachable, or not ours: try to create
        pass
    try:
        client.create_bucket(Bucket=cfg.bucket)
        logger.info("created the content store bucket %s", cfg.bucket)
    except Exception:  # noqa: BLE001 - a store that refuses says so on the first write
        logger.warning(
            "could not create the content store bucket %s; content writes will fail "
            "until it exists",
            cfg.bucket,
            exc_info=True,
        )


def build_sts_client(cfg: ObjectStoreConfig) -> Any:
    """The session-issuing client for an S3-compatible store."""
    import boto3

    return boto3.client(
        "sts",
        endpoint_url=cfg.endpoint_url or None,
        aws_access_key_id=cfg.access_key or None,
        aws_secret_access_key=cfg.secret_key or None,
        region_name=cfg.region,
    )


class StsScopedCredentialMinter:
    """Short-lived S3 sessions, each limited to one scope's prefix."""

    def __init__(self, cfg: ObjectStoreConfig, sts_client: Any) -> None:
        self._cfg = cfg
        self._sts = sts_client

    @property
    def policy_version(self) -> str:
        return "sts-scope-prefix-1"

    def mint(
        self, scope: str, operations: tuple[ContentOperationKind, ...], ttl_sec: float
    ) -> ScopedContentCredential:
        response = self._sts.assume_role(
            RoleArn="arn:x:ignored:for:s3-compatible-sts",
            RoleSessionName=f"fabric-{scope}"[:64],
            Policy=scope_policy(self._cfg.bucket, self._cfg.prefix, scope, operations),
            DurationSeconds=max(900, int(ttl_sec)),
        )
        session = response["Credentials"]
        return ScopedContentCredential(
            material={
                "access_key": session["AccessKeyId"],
                "secret_key": session["SecretAccessKey"],
                "session_token": session["SessionToken"],
            }
        )


class DeploymentCredentialMinter:
    """The deployment's own standing credential, the same for every scope.

    It reaches the whole store, so the store's own policy is the only boundary between
    one scope's content and another's. A deployment opts into that deliberately.
    """

    def __init__(self, cfg: ObjectStoreConfig, logger: logging.Logger) -> None:
        self._cfg = cfg
        logger.warning(
            "content store access is the deployment credential: every task reaches "
            "every scope's content in %s",
            cfg.bucket or cfg.filesystem_root,
        )

    @property
    def policy_version(self) -> str:
        return "deployment-credential-1"

    def mint(
        self, scope: str, operations: tuple[ContentOperationKind, ...], ttl_sec: float
    ) -> ScopedContentCredential:
        return ScopedContentCredential(
            material={
                "access_key": self._cfg.access_key,
                "secret_key": self._cfg.secret_key,
            }
        )

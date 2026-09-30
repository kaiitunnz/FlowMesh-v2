"""A session account's throwaway password hash is generated from a value openssl
reads as a password."""

import subprocess
from typing import Any
from unittest.mock import patch

from worker.executors.ssh_session import session_identity


def test_the_password_openssl_hashes_never_reads_as_an_option() -> None:
    argv_seen: list[list[str]] = []

    def run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
        argv_seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"$6$salt$hash\n")

    with (
        patch.object(session_identity.shutil, "which", return_value="/usr/bin/openssl"),
        patch.object(session_identity, "_run", side_effect=run),
        patch.object(session_identity.secrets, "token_urlsafe", return_value="-Xa9"),
    ):
        assert session_identity._unusable_password_hash() == "$6$salt$hash"

    ((*_, password),) = argv_seen
    assert not password.startswith("-")


def test_an_account_name_is_derived_from_its_session() -> None:
    name: Any = session_identity.account_name_for(
        "ssn-0123456789abcdef0123456789abcdef"
    )

    assert name == "fmssn0123456789abcdef"
    assert session_identity.ACCOUNT_NAME_RE.match(name)

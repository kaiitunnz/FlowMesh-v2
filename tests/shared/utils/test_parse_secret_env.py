import pytest

from shared.utils import parse_secret_env


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_or_unset_reads_as_none(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("SECRET_UNDER_TEST", raising=False)
    else:
        monkeypatch.setenv("SECRET_UNDER_TEST", raw)
    assert parse_secret_env("SECRET_UNDER_TEST") is None


def test_value_is_stripped_and_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_UNDER_TEST", "  s3cret  ")
    secret = parse_secret_env("SECRET_UNDER_TEST")
    assert secret is not None
    assert secret.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(secret)

"""A peer certificate covers a host only through its subject alternative names."""

import pytest

from shared.network.mtls import PeerNames


def _cert(*sans: tuple[str, str], cn: str = "node") -> dict:
    return {"subject": ((("commonName", cn),),), "subjectAltName": sans}


@pytest.mark.parametrize(
    ("sans", "host", "covered"),
    [
        ((("IP Address", "10.0.0.7"),), "10.0.0.7", True),
        ((("IP Address", "10.0.0.7"),), "10.0.0.8", False),
        ((("IP Address", "2001:DB8:0:0:0:0:0:1\n"),), "2001:db8::1", True),
        ((("DNS", "10.0.0.7"),), "10.0.0.7", False),
        ((("DNS", "Node-A.example"),), "node-a.example", True),
        ((("DNS", "*.example"),), "node-a.example", True),
        ((("DNS", "*.example"),), "a.b.example", False),
        ((("DNS", "*.example"),), "example", False),
        ((("URI", "node-a.example"),), "node-a.example", False),
        ((), "node", False),
    ],
)
def test_names_cover_a_host_as_tls_hostname_verification_does(
    sans: tuple[tuple[str, str], ...], host: str, covered: bool
) -> None:
    assert PeerNames.of(_cert(*sans)).covers(host) is covered


def test_a_common_name_never_covers_a_host() -> None:
    names = PeerNames.of(_cert(("DNS", "node-b.example"), cn="10.0.0.7"))
    assert not names.covers("10.0.0.7")
    assert names.covers("node-b.example")

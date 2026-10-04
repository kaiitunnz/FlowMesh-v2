"""Stack env files parse as Docker Compose reads them.

The expectations are what ``docker compose config`` (v5.1.0) resolves for the same
lines, so a value the CLI exports for interpolation matches the value a container's
``env_file`` reader sees.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh_stack.env import load_env, parse_env_file, parse_env_text

_CASES = r"""A=plain
B="double quoted"
C='single quoted'
D="with \"escaped\" quote"
E="line\nbreak"
F='no \n escape'
G=value # comment
H=value#nocomment
I="quoted" # comment
J=  spaced  
K=${A}-ref
L="${A}-ref"
M='${A}-ref'
export N=exported
O=  "lead space quoted"
P=
Q="multi
line"
R=back\\slash
S="a\\b"
T="tab\there"
U=un"quoted"mid
W="dollar \$A"
X=$A
E1=a$$b
E2=${NOPE:-${A}x}
E4="esc \$A and \\$A"
E5=$NOPE-end
E6=${A}${NOPE}
E7="a 'b' c"
E8='a "b" c'
E9 = spaced key
E10="x" trailing
E11=a\ b
E13=${A:-}
E14=$1abc
D1=${NOPE:-dflt}
D2=${A:+alt}
D3=${NOPE-d3}
D6=a=b=c
D7=x\#y # c
D8= # only comment
"""

_EXPECTED = {
    "A": "plain",
    "B": "double quoted",
    "C": "single quoted",
    "D": 'with "escaped" quote',
    "E": "line\nbreak",
    "F": "no \\n escape",
    "G": "value",
    "H": "value#nocomment",
    "I": "quoted",
    "J": "spaced",
    "K": "plain-ref",
    "L": "plain-ref",
    "M": "${A}-ref",
    "N": "exported",
    "O": "lead space quoted",
    "P": "",
    "Q": "multi\nline",
    "R": "back\\\\slash",
    "S": "a\\b",
    "T": "tab\there",
    "U": 'un"quoted"mid',
    "W": "dollar $A",
    "X": "plain",
    "E1": "a$b",
    "E2": "plainx",
    "E4": "esc $A and \\plain",
    "E5": "-end",
    "E6": "plain",
    "E7": "a 'b' c",
    "E8": 'a "b" c',
    "E9": "spaced key",
    "E10": "x",
    "E11": "a\\ b",
    "E13": "plain",
    "E14": "$1abc",
    "D1": "dflt",
    "D2": "alt",
    "D3": "d3",
    "D6": "a=b=c",
    "D7": "x\\#y",
    "D8": "# only comment",
}


def test_values_parse_as_compose_reads_them() -> None:
    assert parse_env_text(_CASES, {}) == _EXPECTED


def test_interpolation_reads_the_environment_after_the_file() -> None:
    assert parse_env_text("Y=${ZZ}-shell\n", {"ZZ": "zz"}) == {"Y": "zz-shell"}


@pytest.mark.parametrize(
    "text", ['A="unterminated\n', "A=${NOPE:?must be set}\n", "A=${NOPE?}\n"]
)
def test_compose_errors_are_errors(text: str) -> None:
    with pytest.raises(ValueError):
        parse_env_text(text, {})


def test_a_quoted_secret_is_exported_as_the_container_reads_it(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TELEMETRY_OTLP_TOKEN=\"tok'en=1\"\nCH_PASSWORD='p\"w'\n")
    with patch.dict(os.environ, {}, clear=True):
        load_env(env_file)
        assert os.environ["TELEMETRY_OTLP_TOKEN"] == "tok'en=1"
        assert os.environ["CH_PASSWORD"] == 'p"w'
        assert parse_env_file(env_file) == {
            "TELEMETRY_OTLP_TOKEN": "tok'en=1",
            "CH_PASSWORD": 'p"w',
        }

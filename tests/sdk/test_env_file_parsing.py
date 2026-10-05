"""Stack env files follow Docker Compose's dotenv quoting, escape and interpolation
rules.

The expectations are what ``docker compose config`` (v5.1.0) resolves for the same
lines, so a value the CLI exports for interpolation matches the value a container's
``env_file`` reader sees.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from flowmesh_stack.doctor import run_doctor_checks
from flowmesh_stack.env import (
    EnvFileError,
    load_env,
    parse_env_file,
    parse_env_text,
    validate_env_file,
)
from flowmesh_stack.env_schema import EnvSchema

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


# Lines a raw string cannot hold: tabs, a no-break space, and escaped single quotes.
_MORE_CASES = (
    "C1=value\t# c\n"
    "C2=value \t# c\n"
    "C3=value\u00a0# c\n"
    "export\tC4=x\n"
    "C5='it\\'s'\n"
    "C6='x\\'y' # c\n"
)

_MORE_EXPECTED = {
    "C1": "value\t# c",
    "C2": "value \t# c",
    "C3": "value\u00a0# c",
    "C4": "x",
    "C5": "it's",
    "C6": "x'y",
}


def test_values_parse_as_compose_reads_them() -> None:
    assert parse_env_text(_CASES, {}) == _EXPECTED
    assert parse_env_text(_MORE_CASES, {}) == _MORE_EXPECTED


# Each case resolves as `docker compose config` (v5.1.0) does once the stack has
# exported the file's values into its environment.
@pytest.mark.parametrize(
    ("text", "environ", "expected"),
    [
        ("A=${B}\nB=${C}\nC=c\n", {}, {"A": "c", "B": "c", "C": "c"}),
        (
            "A=x${B}\nB=y${C}\nC=z${D}\nD=d\n",
            {},
            {"A": "xyzd", "B": "yzd", "C": "zd", "D": "d"},
        ),
        ("A=${B}\nB=file\n", {"B": "shell"}, {"A": "file", "B": "file"}),
        ("K=1\nA=${K}\nK=2\n", {}, {"K": "2", "A": "2"}),
        ("P=${P:-x}\n", {}, {"P": "x"}),
        ("A=${B:?need B}\nB=b\n", {}, {"A": "b", "B": "b"}),
        ("A=${X:-${B}}\nB=b\n", {}, {"A": "b", "B": "b"}),
    ],
)
def test_a_reference_to_another_key_reads_its_final_value(
    text: str, environ: dict[str, str], expected: dict[str, str]
) -> None:
    assert parse_env_text(text, environ) == expected


def test_a_required_reference_to_an_empty_later_key_is_an_error() -> None:
    with pytest.raises(EnvFileError, match="A: B need B"):
        parse_env_text("A=${B:?need B}\nB=\n", {})


def test_keys_that_reference_each_other_in_a_cycle_are_an_error() -> None:
    with pytest.raises(EnvFileError, match="A -> B -> C -> A"):
        parse_env_text("A=${B}\nB=x${C}\nC=${A}\nD=d\n", {})


def test_a_long_reference_chain_resolves_in_linear_time() -> None:
    count = 5000
    text = "".join(f"K{i}=${{K{i + 1}}}\n" for i in range(count)) + f"K{count}=end\n"
    start = time.perf_counter()
    values = parse_env_text(text, {})
    elapsed = time.perf_counter() - start
    assert values["K0"] == "end" and len(values) == count + 1
    assert elapsed < 1.0


def test_a_key_reads_its_own_name_from_the_environment() -> None:
    assert parse_env_text("P=$P:/x\n", {"P": "/bin"}) == {"P": "/bin:/x"}


def test_interpolation_reads_the_environment_after_the_file() -> None:
    assert parse_env_text("Y=${ZZ}-shell\n", {"ZZ": "zz"}) == {"Y": "zz-shell"}


@pytest.mark.parametrize(
    "text",
    [
        'A="unterminated\n',
        "A='ends in an escaped quote\\'\n",
        "A=${NOPE:?must be set}\n",
        "A=${NOPE?}\n",
    ],
)
def test_compose_errors_are_errors(text: str) -> None:
    with pytest.raises(EnvFileError):
        parse_env_text(text, {})


def test_the_doctor_reports_a_malformed_file_as_a_finding(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('REDIS_PASSWORD="abc\n')

    values, errors = validate_env_file(env_file, expected_keys={"REDIS_PASSWORD"})
    assert values is None
    assert errors == [f"{env_file}: REDIS_PASSWORD: unterminated quoted value"]

    report = run_doctor_checks(env_file, EnvSchema(name="t", header=[], sections=[]))
    assert any("unterminated quoted value" in f.message for f in report.findings)


def test_the_doctor_reads_keys_without_interpolating_them(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=${HOME:?HOME must be set}\n")
    with patch.dict(os.environ, {"HOME": "/home/user"}):
        values, errors = validate_env_file(env_file, expected_keys={"A"})
    assert errors == []
    assert values == {"A": "/home/user"}


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

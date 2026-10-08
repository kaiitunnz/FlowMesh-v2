"""Only a store that cannot take writes for now holds a transition's write."""

import pytest
from redis.exceptions import (
    BusyLoadingError,
    ClusterDownError,
    ConnectionError,
    DataError,
    ExecAbortError,
    MasterDownError,
    OutOfMemoryError,
    ReadOnlyError,
    ResponseError,
    TimeoutError,
    TryAgainError,
)

from server.task.runtime.commits import store_unavailable


def _queued(reply: str) -> ResponseError:
    return ResponseError(f"Command # 1 (SET a 1) of pipeline caused error: {reply}")


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out"),
        OSError("network unreachable"),
        ReadOnlyError("You can't write against a read only replica."),
        BusyLoadingError("Redis is loading the dataset in memory"),
        OutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
        TryAgainError("Multiple keys request during rehashing of slot"),
        ClusterDownError("The cluster is down"),
        MasterDownError("Link with MASTER is down"),
        ResponseError("NOREPLICAS Not enough good replicas to write."),
        ResponseError("BUSY Redis is busy running a script."),
        _queued("NOREPLICAS Not enough good replicas to write."),
        _queued("READONLY You can't write against a read only replica."),
    ],
    ids=type,
)
def test_a_store_that_cannot_take_writes_for_now_holds_the_write(
    error: BaseException,
) -> None:
    assert store_unavailable(error)


@pytest.mark.parametrize(
    "error",
    [
        DataError("Invalid input of type: 'NoneType'."),
        _queued("WRONGTYPE Operation against a key holding the wrong kind of value"),
        ResponseError("ERR unknown command 'NOSUCHCMD'"),
        ExecAbortError("Transaction discarded because of previous errors."),
        TypeError("unserializable record"),
        ValueError("bad value"),
    ],
    ids=type,
)
def test_any_other_write_error_is_a_fault_in_the_transition(
    error: BaseException,
) -> None:
    assert not store_unavailable(error)

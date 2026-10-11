"""A failed retirement is retried on the control loop with capped backoff until it
lands."""

import asyncio
import logging

from server.resident.retirement import RetirementRetry


class _Flaky:
    """Fails each key a set number of times, then succeeds."""

    def __init__(self, failures: dict[tuple[str, str], int]) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str]] = []

    def __call__(self, kind: str, key: str) -> None:
        self.calls.append((kind, key))
        if self.failures.get((kind, key), 0) > 0:
            self.failures[(kind, key)] -= 1
            raise ConnectionError("store unavailable")


def _retry(retire: _Flaky, **kwargs: float) -> RetirementRetry:
    return RetirementRetry(retire, logging.getLogger("test"), **kwargs)


def test_a_retry_that_lands_leaves_nothing_pending() -> None:
    retire = _Flaky({})
    retry = _retry(retire)
    retry.add("request", "inv-1")

    retry.run()

    assert retire.calls == [("request", "inv-1")]
    assert retry.pending == []


def test_a_failing_retry_stays_pending_and_backs_off_to_its_cap() -> None:
    retire = _Flaky({("replica", "rpl-1"): 10})
    retry = _retry(retire, base_delay_sec=1.0, max_delay_sec=4.0)
    retry.add("replica", "rpl-1")

    delays = []
    for _ in range(4):
        retry.run()
        delays.append(retry._delay)

    assert retry.pending == [("replica", "rpl-1")]
    assert delays == [2.0, 4.0, 4.0, 4.0]


def test_the_backoff_resets_once_every_retry_lands() -> None:
    retire = _Flaky({("family", "fam"): 2})
    retry = _retry(retire, base_delay_sec=1.0)
    retry.add("family", "fam")
    retry.run()
    retry.run()

    retry.run()

    assert retry.pending == []
    assert retry._delay == 1.0


def test_a_bound_loop_runs_the_retries_without_being_asked() -> None:
    retire = _Flaky({("request", "inv-1"): 1})

    async def scenario() -> list[tuple[str, str]]:
        retry = _retry(retire, base_delay_sec=0.01, max_delay_sec=0.02)
        retry.bind_loop(asyncio.get_running_loop())
        retry.add("request", "inv-1")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if not retry.pending:
                break
        return retry.pending

    assert asyncio.run(scenario()) == []
    assert retire.calls == [("request", "inv-1"), ("request", "inv-1")]


def test_a_cancelled_retry_runs_nothing() -> None:
    retire = _Flaky({})

    async def scenario() -> None:
        retry = _retry(retire, base_delay_sec=0.01)
        retry.bind_loop(asyncio.get_running_loop())
        retry.add("request", "inv-1")
        retry.cancel()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert retire.calls == []

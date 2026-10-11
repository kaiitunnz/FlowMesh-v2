"""A write a store failure interrupted is retried on the control loop with capped
backoff until it lands."""

import asyncio
import logging

from server.resident.store_retry import StoreRetry


class _Flaky:
    """An attempt that fails a set number of times, then lands."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("store unavailable")


def _retry(**kwargs: float) -> StoreRetry:
    return StoreRetry(logging.getLogger("test"), **kwargs)


def test_a_retry_that_lands_leaves_nothing_pending() -> None:
    attempt = _Flaky()
    retry = _retry()
    retry.add("request", "inv-1", attempt)

    retry.run()

    assert attempt.calls == 1
    assert retry.pending == []


def test_a_failing_retry_stays_pending_and_backs_off_to_its_cap() -> None:
    retry = _retry(base_delay_sec=1.0, max_delay_sec=4.0)
    retry.add("replica", "rpl-1", _Flaky(10))

    delays = []
    for _ in range(4):
        retry.run()
        delays.append(retry._delay)

    assert retry.pending == [("replica", "rpl-1")]
    assert delays == [2.0, 4.0, 4.0, 4.0]


def test_an_attempt_that_hands_its_entry_back_counts_as_failed() -> None:
    retry = _retry(base_delay_sec=1.0)

    def handed_back() -> None:
        retry.add("replica", "rpl-1", handed_back)

    retry.add("replica", "rpl-1", handed_back)
    retry.run()

    assert retry.pending == [("replica", "rpl-1")]
    assert retry._delay == 2.0


def test_one_failing_entry_does_not_block_the_others() -> None:
    landed = _Flaky()
    retry = _retry()
    retry.add("family", "stuck", _Flaky(10))
    retry.add("settlement", "inv-1", landed)

    retry.run()

    assert landed.calls == 1
    assert retry.pending == [("family", "stuck")]


def test_the_backoff_resets_once_every_retry_lands() -> None:
    retry = _retry(base_delay_sec=1.0)
    retry.add("family", "fam", _Flaky(2))
    retry.run()
    retry.run()

    retry.run()

    assert retry.pending == []
    assert retry._delay == 1.0


def test_a_bound_loop_runs_the_retries_without_being_asked() -> None:
    attempt = _Flaky(1)

    async def scenario() -> list[tuple[str, str]]:
        retry = _retry(base_delay_sec=0.01, max_delay_sec=0.02)
        retry.bind_loop(asyncio.get_running_loop())
        retry.add("request", "inv-1", attempt)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if not retry.pending:
                break
        return retry.pending

    assert asyncio.run(scenario()) == []
    assert attempt.calls == 2


def test_a_cancelled_retry_runs_nothing() -> None:
    attempt = _Flaky()

    async def scenario() -> None:
        retry = _retry(base_delay_sec=0.01)
        retry.bind_loop(asyncio.get_running_loop())
        retry.add("request", "inv-1", attempt)
        retry.cancel()
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert attempt.calls == 0

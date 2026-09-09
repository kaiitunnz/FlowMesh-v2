"""The bounded response channel records a dropped frame rather than losing it silently.

A client that stops draining cannot pin unbounded memory, so a frame past the bound is
dropped — but the channel marks the loss so its connection is aborted at the terminal
instead of closing cleanly and implying a response that kept every byte.
"""

from worker.serve_ingress.channel import _QUEUE_MAX, ServeIngressChannel


def test_a_full_channel_records_the_loss_and_still_lands_the_terminal() -> None:
    channel = ServeIngressChannel()
    channel.head(200, ())
    # Overflow the bound with body frames a stalled client never drains.
    for _ in range(_QUEUE_MAX + 10):
        channel.chunk(b"x")
    assert channel.lost is True

    channel.complete()
    # The terminal always reaches the connection so the response closes.
    seen_terminal = False
    while (frame := channel.drain(0.01)) is not None:
        if frame.terminal:
            seen_terminal = True
    assert seen_terminal is True


def test_a_channel_that_keeps_every_frame_reports_no_loss() -> None:
    channel = ServeIngressChannel()
    channel.head(200, ())
    channel.chunk(b"a")
    channel.chunk(b"b")
    channel.complete()
    assert channel.lost is False

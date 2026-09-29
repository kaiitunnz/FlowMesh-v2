import logging
import os

logger = logging.getLogger(__name__)


def signal_process_group(pgid: int, sig: int) -> None:
    """Send ``sig`` to a child's own process group.

    Group 1 and the caller's own group are refused: signalling either reaches far
    more than the child, since ``killpg(1)`` is ``kill(-1)``, every process the caller
    may signal.
    """
    pgid = int(pgid)
    if pgid <= 1 or pgid == os.getpgrp():
        logger.error("Refusing to signal process group %d", pgid)
        return
    os.killpg(pgid, sig)

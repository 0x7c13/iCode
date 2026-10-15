# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Closing the terminal (SIGHUP) or SIGTERM ends iCode through its normal teardown.

Unhandled, either signal ends the process at once, and the session's background
commands — in process sessions of their own, so the terminal's SIGHUP never
reaches them — keep running with nobody left to stop them. Windows has neither
signal.
"""

from __future__ import annotations

import signal
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

HANGUP_SIGNALS: tuple[int, ...] = () if sys.platform == "win32" else (signal.SIGHUP, signal.SIGTERM)


@contextmanager
def hangup_as_interrupt() -> Iterator[None]:
    """While inside, SIGHUP and SIGTERM act as Ctrl+C: ``asyncio.run`` cancels its main task, so teardown runs.

    A process started with Ctrl+C ignored, as ``&`` in a script does, keeps the
    default: the signals still end it. A signal it was started ignoring, as
    ``nohup`` does with SIGHUP, stays ignored.
    """
    if (
        not HANGUP_SIGNALS
        or threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGINT) is not signal.default_int_handler
    ):
        yield
        return
    previous = {
        signum: signal.signal(signum, _interrupt)
        for signum in HANGUP_SIGNALS
        if signal.getsignal(signum) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _interrupt(_signum: int, _frame: FrameType | None) -> None:
    signal.raise_signal(signal.SIGINT)

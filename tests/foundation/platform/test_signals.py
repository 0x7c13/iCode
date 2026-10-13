# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Closing the terminal or SIGTERM ends an ``asyncio.run`` through its teardown, as Ctrl+C does."""

from __future__ import annotations

import asyncio
import signal
import sys

import pytest

from chrys.foundation.platform.signals import HANGUP_SIGNALS, hangup_as_interrupt

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Windows has neither signal")


@pytest.mark.parametrize("name", ["SIGHUP", "SIGTERM"])
def test_the_signal_cancels_the_main_task_so_its_teardown_runs(name: str) -> None:
    signum = getattr(signal, name)
    assert signum in HANGUP_SIGNALS
    torn_down: list[str] = []

    async def main() -> None:
        try:
            signal.raise_signal(signum)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            torn_down.append("cancelled")
            raise

    # asyncio.run turns Ctrl+C into a cancel only while Python's own SIGINT handler is installed.
    interrupt = signal.signal(signal.SIGINT, signal.default_int_handler)
    # As a terminal starts a process: a CI runner may start the tests with SIGHUP ignored, as nohup does.
    hangup = signal.signal(signum, signal.SIG_DFL)
    try:
        with pytest.raises(KeyboardInterrupt), hangup_as_interrupt():
            asyncio.run(main())
        assert signal.getsignal(signum) is signal.SIG_DFL
    finally:
        signal.signal(signum, hangup)
        signal.signal(signal.SIGINT, interrupt)

    assert torn_down == ["cancelled"]


def test_a_process_started_with_ctrl_c_ignored_keeps_the_default_for_both_signals() -> None:
    # As ``icode run … &`` in a script starts it: there Ctrl+C would never stop it, and neither would SIGTERM.
    interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
    before = {signum: signal.getsignal(signum) for signum in HANGUP_SIGNALS}
    try:
        with hangup_as_interrupt():
            assert {signum: signal.getsignal(signum) for signum in HANGUP_SIGNALS} == before
    finally:
        signal.signal(signal.SIGINT, interrupt)


def test_a_signal_the_process_was_started_ignoring_stays_ignored() -> None:
    # As ``nohup icode run …`` starts it: closing the terminal must not end the run.
    interrupt = signal.signal(signal.SIGINT, signal.default_int_handler)
    hangup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    terminate = signal.getsignal(signal.SIGTERM)
    try:
        with hangup_as_interrupt():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is not terminate
        assert signal.getsignal(signal.SIGTERM) is terminate
    finally:
        signal.signal(signal.SIGHUP, hangup)
        signal.signal(signal.SIGINT, interrupt)

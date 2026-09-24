# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Keep a job-control stopped child from wedging the asyncio event loop on macOS.

CPython 3.14.7 moved subprocess reaping onto the event loop thread: a per-child
waiter thread parks in ``waitid(P_PID, pid, WEXITED | WNOWAIT)`` and, once that
returns, schedules ``_reap_and_notify`` — a blocking ``waitpid(pid, 0)`` — on the
loop, so the reap and the returncode notification are atomic with respect to it.

Darwin returns from that ``waitid`` when the child merely *stops*: the result
carries ``si_code == CLD_STOPPED``, which upstream discards.  The loop thread
then blocks in ``waitpid(pid, 0)``, which cannot return until the child exits,
and a stopped child does not exit on its own.  The whole event loop is wedged —
including :func:`chrys.foundation.platform.process.wait_for_subprocess`, whose
stopped-process detection exists to kill exactly this child.  A shell command
that stops (``kill -STOP $$``, or a program running its own job control) would
otherwise hang the entire TUI, not just the tool call.

Check the reported state in the waiter thread and schedule reaping only after
an exit.  Keep upstream's atomic loop-thread reap and notification: reaping in
the waiter would free the PID while ``Process.send_signal()`` still considers
it live, allowing cancellation to signal an unrelated process that reused it.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_STOPPED_CHILD_POLL_INTERVAL = 0.05

_REQUIRED_ATTRIBUTES = ("_do_waitpid", "_reap", "_reap_and_notify")
"""Shape of the upstream implementation this shim replaces a method on.

``_reap_and_notify`` is the atomic loop-thread reap the shim must preserve: an
implementation without it does not hand a blocking ``waitpid`` to the loop and
must be left alone.  ``_threads`` is deliberately absent — it is set in
``__init__``, so it is not visible on the class.
"""


def _do_waitpid_until_exit(
    self: Any,
    loop: asyncio.AbstractEventLoop,
    expected_pid: int,
    callback: Callable[..., object],
    args: tuple[object, ...],
) -> None:
    """Leave the PID reserved until the loop can reap and notify atomically."""
    # ``waitid`` and its constants are POSIX-only, and this runs on Darwin alone.
    posix_os = cast(Any, os)
    exited = (posix_os.CLD_EXITED, posix_os.CLD_KILLED, posix_os.CLD_DUMPED)
    try:
        try:
            while True:
                result = posix_os.waitid(posix_os.P_PID, expected_pid, posix_os.WEXITED | posix_os.WNOWAIT)
                if result is not None and result.si_code in exited:
                    break
                # Darwin can repeatedly report the same unconsumed stopped
                # state. Back off in this child-owned waiter thread, never on
                # the loop that detects and terminates stopped processes.
                time.sleep(_STOPPED_CHILD_POLL_INTERVAL)
        except ChildProcessError:
            # Preserve upstream's handling of a child reaped elsewhere.
            pass
        if not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._reap_and_notify, loop, expected_pid, callback, args)
                return
            except RuntimeError:
                # The loop closed between the check and the scheduling call.
                pass
        pid, _ = self._reap(loop, expected_pid)
        logger.warning("Loop %r that handles pid %r is closed", loop, pid)
    finally:
        self._threads.pop(expected_pid, None)


def install_stopped_child_reap_fix() -> bool:
    """Apply the shim on affected platforms; report whether it was applied.

    Idempotent, and a no-op wherever the upstream implementation does not have
    the shape described in the module docstring — including every non-Darwin
    platform and any CPython that reaps off the event loop thread already.
    """
    if sys.platform != "darwin":
        return False

    from asyncio import unix_events

    watcher = getattr(unix_events, "_ThreadedChildWatcher", None)
    if watcher is None or not all(hasattr(watcher, name) for name in _REQUIRED_ATTRIBUTES):
        return False
    if watcher._do_waitpid is _do_waitpid_until_exit:
        return False

    watcher._do_waitpid = _do_waitpid_until_exit
    return True

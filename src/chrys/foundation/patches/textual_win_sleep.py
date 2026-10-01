# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Sleep Textual's Windows timers on the event loop's completion port, not on executor threads.

Textual 8.2.7 sleeps every timer tick in ``run_in_executor(None, WaitForMultipleObjects)``:
each armed timer pins a default-executor thread for the length of its sleep and every tick
is a thread hop each way. A screen with a handful of animations fills the small default
pool (``min(32, cpus + 4)`` threads), so ``to_thread`` work queues behind sleeping timers
and the loop thread stalls in ``Thread.start`` growing the pool. Its cancellation also
closes the timer handles without joining the native wait.

Keep Textual's high-resolution waitable timer for its accuracy, but register it with the
``ProactorEventLoop``'s I/O completion port — the wait asyncio itself uses for subprocess
exit. A kernel wait thread posts to the port when the timer fires, the loop completes the
future itself, and no Python thread sleeps. Cancellation re-arms the timer to fire at once,
so the registered wait completes and unregisters itself before the handle closes; a port
that stops delivering gets the wait unregistered after a short bound instead of wedging
the cancellation. A loop without a completion port sleeps on the loop's own clock. Both Textual entry points are
replaced, including the alias already imported by ``textual._time``.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
from collections.abc import Callable
from ctypes.wintypes import LARGE_INTEGER
from typing import Any

from chrys.foundation.platform import get_platform

_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_RUNTIME_PATCH_MARKER = "_chrys_registers_windows_timer_wait"
_CREATE_WAITABLE_TIMER_MANUAL_RESET = 0x00000001
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003
# Negative due times are relative, in 100 ns units.
_FIRE_AT_ONCE = LARGE_INTEGER(-1)
# A re-armed timer reaches the port within microseconds; a wait still pending after
# this bound sits on a port that no longer delivers, and cancellation must not hang on it.
_REARM_SETTLE_SECONDS = 1.0
logger = logging.getLogger(__name__)

type HandleWait = Callable[[int], asyncio.Future[bool]]


def _completion_port_wait(loop: asyncio.AbstractEventLoop) -> HandleWait | None:
    """The loop's registered-handle wait, when the loop runs on an I/O completion port.

    ``ProactorEventLoop`` exposes it as ``IocpProactor.wait_for_handle``: the kernel's
    wait threads post to the port when the handle is signalled and the loop completes
    the returned future itself, with no thread of ours blocked.
    """
    proactor = getattr(loop, "_proactor", None)
    wait_for_handle = getattr(proactor, "wait_for_handle", None)
    return wait_for_handle if callable(wait_for_handle) else None


async def _await_timer(timer: int, wait_for_handle: HandleWait, *, kernel32: Any) -> bool:
    """Await the timer through the completion port; return whether that wait completed.

    A cancelled caller re-arms the timer to fire at once: the registered wait then
    completes through the port and unregisters itself, and only after that does the
    caller close the handle. A failed re-arm, or a port that never delivers the
    re-armed timer, falls back to unregistering the pending wait, which the proactor
    does synchronously in ``cancel()``.
    """
    try:
        waiter = wait_for_handle(timer)
    except OSError:
        return False
    try:
        await asyncio.shield(waiter)
    except asyncio.CancelledError:
        if not waiter.done():
            if kernel32.SetWaitableTimer(timer, ctypes.byref(_FIRE_AT_ONCE), 0, None, None, 0):
                await _settle_registered_wait(waiter)
            else:
                waiter.cancel()
        raise
    except OSError:
        return False
    return True


async def _settle_registered_wait(waiter: asyncio.Future[bool]) -> None:
    """Let the re-armed timer complete its registered wait, unregistering it past the bound."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _REARM_SETTLE_SECONDS
    while not waiter.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            waiter.cancel()
            return
        # Further cancellation of the caller changes nothing: the handle stays open
        # until the wait is off the port, so keep waiting out the same bound.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait({waiter}, timeout=remaining)


async def _sleep(secs: float, *, kernel32: Any) -> None:
    """Sleep on a native high-resolution timer whose handle is owned by this coroutine."""
    # Keep the timing compensation of pinned Textual 8.2.7.
    sleep_for = max(0, secs - 0.001)
    if sleep_for < 0.0005:
        return
    wait_for_handle = _completion_port_wait(asyncio.get_running_loop())
    if wait_for_handle is None:
        await asyncio.sleep(sleep_for)
        return

    # Manual reset: a completed wait leaves the timer signalled, which the port's
    # post-wait poll expects of a handle (a synchronization timer reads as never fired).
    timer = kernel32.CreateWaitableTimerExW(
        None,
        None,
        _CREATE_WAITABLE_TIMER_MANUAL_RESET | _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
        _TIMER_ALL_ACCESS,
    )
    if not timer:
        await asyncio.sleep(sleep_for)
        return
    fired = False
    try:
        due = LARGE_INTEGER(int(sleep_for * -10_000_000))
        if kernel32.SetWaitableTimer(timer, ctypes.byref(due), 0, None, None, 0):
            fired = await _await_timer(timer, wait_for_handle, kernel32=kernel32)
    finally:
        # The registered wait has completed or been unregistered, so nothing waits on the
        # handle any more. Allocation happens after coroutine entry, so cancellation before
        # the task starts leaks nothing.
        kernel32.CloseHandle(timer)
    if not fired:
        await asyncio.sleep(sleep_for)


def apply_runtime_patch() -> None:
    """Patch the Windows timer before any Textual timers start."""
    if not get_platform().is_windows:
        return
    try:
        import textual
        import textual._time as time_mod
        import textual._win_sleep as win_sleep_mod
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning("Skipping Windows timer patch for unsupported Textual %s", textual.__version__)
        return
    if not hasattr(win_sleep_mod, "kernel32"):
        # Textual itself falls back to asyncio.sleep when Win32 is unavailable.
        return
    if getattr(win_sleep_mod.sleep, _RUNTIME_PATCH_MARKER, False):
        time_mod.win_sleep = win_sleep_mod.sleep
        return
    kernel32 = win_sleep_mod.kernel32

    async def sleep(secs: float) -> None:
        await _sleep(secs, kernel32=kernel32)

    setattr(sleep, _RUNTIME_PATCH_MARKER, True)
    win_sleep_mod.sleep = sleep
    time_mod.win_sleep = sleep

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Windows timer sleeps register with the loop's completion port: no executor thread, handles closed after the wait."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import dataclasses
import importlib.util
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import textual
import textual._time as time_mod

from chrys.foundation.patches import textual_win_sleep as patch
from chrys.foundation.platform import get_platform
from tests.support.waiting import wait_for

_TIMER = 101
_MANUAL_RESET_HIGH_RESOLUTION = 0x3
_TIMER_ALL_ACCESS = 0x1F0003


class _Kernel32:
    """Fake Win32 timer API: records arming and closing, and fails on request."""

    def __init__(self, *, failure: str = "") -> None:
        self.failure = failure
        self.allocations = 0
        self.armed: list[int] = []
        self.closed: list[int] = []

    def CreateWaitableTimerExW(self, attributes, name, flags, access):
        # The port's post-wait poll reads a signalled handle, so the timer must stay signalled.
        assert (flags, access) == (_MANUAL_RESET_HIGH_RESOLUTION, _TIMER_ALL_ACCESS)
        self.allocations += 1
        return 0 if self.failure == "create" else _TIMER

    def SetWaitableTimer(self, timer, due, period, callback, argument, resume):
        assert timer == _TIMER
        if self.failure == "arm" or (self.failure == "rearm" and self.armed):
            return 0
        self.armed.append(due._obj.value)
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


class _CompletionPort:
    """Fake ``IocpProactor.wait_for_handle``: one future per registered handle, completed by the test."""

    def __init__(self, *, failure: str = "") -> None:
        self.failure = failure
        self.registered: list[int] = []
        self.waits: list[asyncio.Future[bool]] = []

    def wait_for_handle(self, handle: int) -> asyncio.Future[bool]:
        if self.failure == "register":
            raise OSError(6, "The handle is invalid.")
        self.registered.append(handle)
        waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.waits.append(waiter)
        return waiter


def _install_native_boundary(monkeypatch: pytest.MonkeyPatch, native: _Kernel32) -> ModuleType:
    """Load pinned Textual's Windows implementation with only its Win32 API replaced."""
    spec = importlib.util.find_spec("textual._win_sleep")
    assert spec is not None and spec.origin is not None
    isolated_spec = importlib.util.spec_from_file_location("textual._win_sleep", spec.origin)
    assert isolated_spec is not None and isolated_spec.loader is not None
    module = importlib.util.module_from_spec(isolated_spec)
    with monkeypatch.context() as loading:
        loading.setattr(ctypes, "windll", SimpleNamespace(kernel32=native), raising=False)
        isolated_spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "textual._win_sleep", module)
    monkeypatch.setattr(textual, "_win_sleep", module, raising=False)
    monkeypatch.setattr(time_mod, "win_sleep", module.sleep, raising=False)
    windows = dataclasses.replace(get_platform(), os_name="windows")
    monkeypatch.setattr(patch, "get_platform", lambda: windows)
    return module


def _install_port(monkeypatch: pytest.MonkeyPatch, port: _CompletionPort | None) -> None:
    """Give the running loop *port* as its completion port (``None``: a loop without one)."""
    monkeypatch.setattr(patch, "_completion_port_wait", lambda _loop: None if port is None else port.wait_for_handle)


def _forbid_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_executor(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a timer sleep must not occupy an executor thread")

    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", no_executor)


def _stub_loop_clock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the patch module's ``asyncio.sleep`` fallback only; the test's own loop keeps the real one."""
    fallback = AsyncMock()
    shadow = ModuleType("asyncio")
    shadow.__dict__.update(vars(asyncio))
    shadow.sleep = fallback
    monkeypatch.setattr(patch, "asyncio", shadow)
    return fallback


def test_completion_port_wait_is_the_proactor_handle_wait() -> None:
    def wait_for_handle(handle: int) -> asyncio.Future[bool]:
        raise NotImplementedError

    assert patch._completion_port_wait(SimpleNamespace(_proactor=SimpleNamespace(wait_for_handle=wait_for_handle))) is (
        wait_for_handle
    )
    assert patch._completion_port_wait(SimpleNamespace()) is None
    assert patch._completion_port_wait(SimpleNamespace(_proactor=None)) is None


@pytest.mark.parametrize("recancel", [False, True])
async def test_cancel_fires_the_timer_and_closes_it_after_the_registered_wait_completes(monkeypatch, recancel):
    native = _Kernel32()
    port = _CompletionPort()
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, port)
    _forbid_executor(monkeypatch)
    patch.apply_runtime_patch()
    task = asyncio.create_task(module.sleep(60))
    try:
        await wait_for(lambda: port.registered == [_TIMER], description="timer registered with the port")
        assert native.armed == [int(59.999 * -10_000_000)]
        task.cancel("timer stopped")
        # Cancellation re-arms the timer to fire at once instead of unregistering the wait.
        await wait_for(lambda: len(native.armed) == 2, description="timer re-armed to fire")
        assert native.armed[-1] == -1
        if recancel:
            task.cancel("runner teardown")
            # The callback runs after the cancellation's queued task wakeup.
            delivered = asyncio.Event()
            asyncio.get_running_loop().call_soon(delivered.set)
            await delivered.wait()
        assert not task.done(), "cancellation returned while the port still waits on the handle"
        assert native.closed == []
        assert not port.waits[0].cancelled()
    finally:
        port.waits[0].set_result(True)
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert task.cancelled()
    assert native.closed == [_TIMER]


async def test_failed_rearm_unregisters_the_pending_wait_instead(monkeypatch):
    native = _Kernel32(failure="rearm")
    port = _CompletionPort()
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, port)
    patch.apply_runtime_patch()
    task = asyncio.create_task(module.sleep(60))
    await wait_for(lambda: port.registered == [_TIMER], description="timer registered with the port")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The proactor unregisters the wait inside cancel(), so the handle closes right after.
    assert port.waits[0].cancelled()
    assert native.closed == [_TIMER]


async def test_a_port_that_never_delivers_is_unregistered_after_the_settle_bound(monkeypatch):
    """The re-armed timer normally completes the wait within microseconds; a dead port must not wedge the cancel."""
    native = _Kernel32()
    port = _CompletionPort()
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, port)
    _forbid_executor(monkeypatch)
    monkeypatch.setattr(patch, "_REARM_SETTLE_SECONDS", 0.05)
    patch.apply_runtime_patch()
    task = asyncio.create_task(module.sleep(60))
    await wait_for(lambda: port.registered == [_TIMER], description="timer registered with the port")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert native.armed[-1] == -1
    assert port.waits[0].cancelled()
    assert native.closed == [_TIMER]


async def test_cancellation_before_start_allocates_no_native_handles(monkeypatch):
    native = _Kernel32()
    port = _CompletionPort()
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, port)
    patch.apply_runtime_patch()
    task = asyncio.create_task(module.sleep(60))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert native.allocations == 0
    assert port.registered == []
    assert native.closed == []


@pytest.mark.parametrize("failure", ["", "create", "arm", "register", "wait"])
async def test_timer_completion_and_native_failure_fallbacks(monkeypatch, failure):
    native = _Kernel32(failure=failure)
    port = _CompletionPort(failure=failure)
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, port)
    _forbid_executor(monkeypatch)
    fallback = _stub_loop_clock(monkeypatch)
    patch.apply_runtime_patch()

    task = asyncio.create_task(module.sleep(1))
    if failure in {"", "wait"}:
        await wait_for(lambda: port.registered == [_TIMER], description="timer registered with the port")
        if failure == "wait":
            port.waits[0].set_exception(OSError(6, "The handle is invalid."))
        else:
            port.waits[0].set_result(True)
    await task

    if failure:
        fallback.assert_awaited_once_with(0.999)
    else:
        fallback.assert_not_awaited()
    assert native.closed == ([] if failure == "create" else [_TIMER])


async def test_a_loop_without_a_completion_port_sleeps_on_the_loop_clock(monkeypatch):
    native = _Kernel32()
    module = _install_native_boundary(monkeypatch, native)
    _install_port(monkeypatch, None)
    _forbid_executor(monkeypatch)
    fallback = _stub_loop_clock(monkeypatch)
    patch.apply_runtime_patch()

    await module.sleep(1)

    fallback.assert_awaited_once_with(0.999)
    assert native.allocations == 0


def test_patch_updates_preimported_alias_and_is_idempotent(monkeypatch):
    native = _Kernel32()
    module = _install_native_boundary(monkeypatch, native)
    original = module.sleep
    assert textual.__version__ == patch._RUNTIME_PATCH_TEXTUAL_VERSION
    patch.apply_runtime_patch()
    installed = module.sleep
    assert installed is not original
    assert time_mod.win_sleep is installed
    patch.apply_runtime_patch()
    assert module.sleep is installed
    assert time_mod.win_sleep is installed


@pytest.mark.parametrize("unsupported", ["platform", "version", "native_api"])
def test_patch_preserves_unsupported_environments(monkeypatch, unsupported):
    native = _Kernel32()
    module = _install_native_boundary(monkeypatch, native)
    original = module.sleep
    if unsupported == "platform":
        other = dataclasses.replace(get_platform(), os_name="linux")
        monkeypatch.setattr(patch, "get_platform", lambda: other)
    elif unsupported == "version":
        monkeypatch.setattr(textual, "__version__", "0.0.0")
    else:
        del module.kernel32
    patch.apply_runtime_patch()
    assert module.sleep is original
    assert time_mod.win_sleep is original


@pytest.mark.skipif(not get_platform().is_windows, reason="requires native Windows timer handles")
def test_real_windows_timers_fire_through_the_port_and_cancel_without_threads():
    # Contain a native regression in a killable process. asyncio.run() must join
    # its default executor on exit, reproducing the Runner.close() boundary.
    script = """
import asyncio
import threading
import time
from chrys.foundation.patches.textual_win_sleep import apply_runtime_patch
apply_runtime_patch()
from textual._time import sleep

async def main():
    started = time.perf_counter()
    await sleep(0.05)
    elapsed = time.perf_counter() - started
    assert 0.03 <= elapsed < 1.0, elapsed
    tasks = [asyncio.create_task(sleep(60)) for _ in range(8)]
    entered = asyncio.Event()
    asyncio.get_running_loop().call_soon(entered.set)
    await entered.wait()
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    executor_threads = [thread.name for thread in threading.enumerate() if thread.name.startswith("asyncio_")]
    assert executor_threads == [], executor_threads

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr

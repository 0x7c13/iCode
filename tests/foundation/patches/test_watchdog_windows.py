# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Windows emitter shutdown must not close a reused native handle."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType

import pytest
import watchdog
import watchdog.observers
import watchdog.version
from watchdog.observers.api import EventQueue, ObservedWatch

from chrys.foundation.patches import watchdog_windows as patch
from chrys.foundation.platform import get_platform


class _NativeDirectory:
    def __init__(self) -> None:
        self.owner = ""
        self.closed: list[int] = []
        self.entered = Event()
        self.release = Event()
        self.release.set()

    def get_directory_handle(self, path: str) -> int:
        self.owner = path
        return 42

    def close_directory_handle(self, handle: int) -> None:
        self.closed.append(handle)
        self.owner = "unrelated native resource"
        self.entered.set()
        assert self.release.wait(5), "test did not release the native close"

    def read_events(self, handle: int, path: str, *, recursive: bool) -> list:
        return []


def _load_emitter(monkeypatch: pytest.MonkeyPatch, native: _NativeDirectory) -> ModuleType:
    """Execute the pinned Windows implementation with only its native API replaced."""
    source = Path(watchdog.__file__).parent / "observers" / "read_directory_changes.py"
    spec = importlib.util.spec_from_file_location("watchdog.observers.read_directory_changes", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    winapi = ModuleType("watchdog.observers.winapi")
    winapi.get_directory_handle = native.get_directory_handle
    winapi.close_directory_handle = native.close_directory_handle
    winapi.read_events = native.read_events
    with monkeypatch.context() as loading:
        loading.setitem(sys.modules, "watchdog.observers.winapi", winapi)
        spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "watchdog.observers.read_directory_changes", module)
    monkeypatch.setattr(watchdog.observers, "read_directory_changes", module, raising=False)
    windows = dataclasses.replace(get_platform(), os_name="windows")
    monkeypatch.setattr(patch, "get_platform", lambda: windows)
    return module


def test_repeated_stop_does_not_close_a_reused_handle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native = _NativeDirectory()
    module = _load_emitter(monkeypatch, native)
    patch.apply_runtime_patch()
    installed = module.WindowsApiEmitter.on_thread_stop
    patch.apply_runtime_patch()
    assert module.WindowsApiEmitter.on_thread_stop is installed
    emitter = module.WindowsApiEmitter(EventQueue(), ObservedWatch(str(tmp_path), recursive=False))
    emitter.on_thread_start()

    # The deleted-directory path calls stop while holding this non-reentrant lock.
    with emitter._lock:
        emitter.stop()
    assert native.owner == "unrelated native resource"
    emitter.stop()  # The observer later stops the same emitter during workspace switch.
    assert native.closed == [42]
    assert emitter._whandle is None


def test_concurrent_stop_detaches_ownership_before_native_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = _NativeDirectory()
    module = _load_emitter(monkeypatch, native)
    patch.apply_runtime_patch()
    emitter = module.WindowsApiEmitter(EventQueue(), ObservedWatch(str(tmp_path), recursive=False))
    emitter.on_thread_start()
    native.release.clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(emitter.stop)
        try:
            assert native.entered.wait(5)
            second = executor.submit(emitter.stop)
            second.result(timeout=5)
            assert native.closed == [42]
            assert emitter._whandle is None
        finally:
            native.release.set()
            first.result(timeout=5)


def test_patch_skips_other_platforms_and_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_emitter(monkeypatch, _NativeDirectory())
    original = module.WindowsApiEmitter.on_thread_stop
    other_platform = dataclasses.replace(get_platform(), os_name="linux")
    with monkeypatch.context() as other:
        other.setattr(patch, "get_platform", lambda: other_platform)
        patch.apply_runtime_patch()
    assert module.WindowsApiEmitter.on_thread_stop is original
    monkeypatch.setattr(watchdog.version, "VERSION_STRING", "999.0.0")
    patch.apply_runtime_patch()
    assert module.WindowsApiEmitter.on_thread_stop is original

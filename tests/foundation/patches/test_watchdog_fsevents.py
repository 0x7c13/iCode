# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A stopped macOS emitter must not stay alive through the callback FSEvents keeps."""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import watchdog
import watchdog.observers
import watchdog.version
from watchdog.events import FileSystemEventHandler
from watchdog.observers.api import EventQueue, ObservedWatch

from chrys.foundation.patches import watchdog_fsevents as patch
from chrys.foundation.platform import get_platform


class _NativeFSEvents:
    """The extension's API; like watchdog 6.0.0's, it never lets go of a callback."""

    def __init__(self) -> None:
        self.callbacks: list[Callable[[list[bytes], list[int], list[int], list[int]], None]] = []

    def add_watch(
        self,
        emitter: object,
        watch: object,
        callback: Callable[[list[bytes], list[int], list[int], list[int]], None],
        paths: list[str],
    ) -> None:
        self.callbacks.append(callback)

    def read_events(self, emitter: object) -> None:
        return None

    def remove_watch(self, watch: object) -> None:
        return None

    def stop(self, emitter: object) -> None:
        return None


def _load_emitter(monkeypatch: pytest.MonkeyPatch, native: _NativeFSEvents) -> ModuleType:
    """Execute the pinned macOS implementation with only its native extension replaced."""
    source = Path(watchdog.__file__).parent / "observers" / "fsevents.py"
    spec = importlib.util.spec_from_file_location("watchdog.observers.fsevents", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    extension = ModuleType("_watchdog_fsevents")
    vars(extension).update(
        add_watch=native.add_watch,
        read_events=native.read_events,
        remove_watch=native.remove_watch,
        stop=native.stop,
    )
    with monkeypatch.context() as loading:
        loading.setitem(sys.modules, "_watchdog_fsevents", extension)
        spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "watchdog.observers.fsevents", module)
    monkeypatch.setattr(watchdog.observers, "fsevents", module, raising=False)
    macos = dataclasses.replace(get_platform(), os_name="macos")
    monkeypatch.setattr(patch, "get_platform", lambda: macos)
    return module


def test_the_extension_keeps_a_forwarder_that_lets_the_emitter_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = _NativeFSEvents()
    module = _load_emitter(monkeypatch, native)
    patch.apply_runtime_patch()
    installed = module.FSEventsEmitter.run
    patch.apply_runtime_patch()
    assert module.FSEventsEmitter.run is installed

    class _Recording(module.FSEventsEmitter):
        received: list[list[bytes]]

        def events_callback(self, paths: list[bytes], inodes: list[int], flags: list[int], ids: list[int]) -> None:
            self.received.append(paths)

    emitter = _Recording(EventQueue(), ObservedWatch(str(tmp_path), recursive=False))
    emitter.received = []
    emitter.run()
    [forward] = native.callbacks
    forward([b"HEAD"], [1], [0], [7])
    assert emitter.received == [[b"HEAD"]]

    collected = weakref.ref(emitter)
    del emitter
    gc.collect()
    assert collected() is None
    forward([b"HEAD"], [1], [0], [8])  # A late delivery after collection is dropped, not raised.


def test_patch_skips_other_platforms_and_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_emitter(monkeypatch, _NativeFSEvents())
    original = module.FSEventsEmitter.run
    other_platform = dataclasses.replace(get_platform(), os_name="linux")
    with monkeypatch.context() as other:
        other.setattr(patch, "get_platform", lambda: other_platform)
        patch.apply_runtime_patch()
    assert module.FSEventsEmitter.run is original
    monkeypatch.setattr(watchdog.version, "VERSION_STRING", "999.0.0")
    patch.apply_runtime_patch()
    assert module.FSEventsEmitter.run is original


class _StandInStderr:
    """What ``sys.stderr`` is while a Textual App runs: an object holding the App."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


@pytest.mark.skipif(not get_platform().is_macos, reason="requires the native FSEvents extension")
def test_a_stopped_observer_releases_what_its_emitter_thread_captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watchdog.observers.fsevents import FSEventsObserver

    stderr = _StandInStderr()
    observer = FSEventsObserver()
    with monkeypatch.context() as capturing:
        capturing.setattr(sys, "stderr", stderr)
        observer.schedule(FileSystemEventHandler(), str(tmp_path), recursive=False)
        observer.start()
    observer.stop()
    observer.join(timeout=5)
    assert not observer.is_alive()

    captured = weakref.ref(stderr)
    del stderr, observer
    gc.collect()
    assert captured() is None

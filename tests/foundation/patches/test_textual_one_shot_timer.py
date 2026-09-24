# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that lets a one-shot Textual timer fire late instead of never."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from textual import _time
from textual import timer as timer_module
from textual.app import App
from textual.timer import Timer

from chrys.foundation.patches.textual_one_shot_timer import apply_runtime_patch
from tests.support.waiting import wait_for

_PROBE = "one-shot-probe"


def _stall_after_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move the probe timer's clock on by five intervals after it reads its start time.

    That is what the timer sees when a garbage collection or the OS stops the process between
    its first two reads; every other clock reader keeps the real time.
    """
    reads = 0

    def get_time() -> float:
        nonlocal reads
        task = asyncio.current_task()
        if task is None or task.get_name() != _PROBE:
            return _time.get_time()
        reads += 1
        return _time.get_time() + (0.1 if reads > 1 else 0.0)

    monkeypatch.setattr(timer_module, "_time", SimpleNamespace(get_time=get_time))


async def test_upstream_never_fires_a_one_shot_timer_stalled_after_it_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    monkeypatch.setattr(Timer, "__init__", inspect.unwrap(Timer.__init__))
    _stall_after_start(monkeypatch)
    fired: list[bool] = []
    app = App()
    async with app.run_test() as pilot:
        timer = app.set_timer(0.02, lambda: fired.append(True), name=_PROBE)
        task = timer._task
        assert task is not None
        await wait_for(task.done, pilot=pilot, description="the timer's task ends")
        # A tick would have queued the callback on the app; drain it.
        await pilot.pause()
        assert fired == []


async def test_a_one_shot_timer_stalled_after_it_starts_still_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_runtime_patch()
    _stall_after_start(monkeypatch)
    fired: list[bool] = []
    app = App()
    async with app.run_test() as pilot:
        app.set_timer(0.02, lambda: fired.append(True), name=_PROBE)
        await wait_for(lambda: fired, pilot=pilot, description="the stalled timer fires")
        assert fired == [True]


def test_repeating_timers_keep_skipping_missed_ticks() -> None:
    apply_runtime_patch()
    app = App()

    assert Timer(app, 0.1, repeat=0)._skip is False
    assert Timer(app, 0.1)._skip is True
    assert Timer(app, 0.1, repeat=3)._skip is True
    assert Timer(app, 0.1, repeat=0, skip=True)._skip is False


def test_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    patched = Timer.__init__

    apply_runtime_patch()

    assert Timer.__init__ is patched
    assert inspect.unwrap(Timer.__init__) is not patched

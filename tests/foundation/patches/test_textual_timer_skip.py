# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that makes a late Textual timer tick skip only the ticks that have passed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import timer as timer_module
from textual.timer import EventTargetGone, Timer

from chrys.foundation.patches import textual_timer_skip
from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.staged_members import install_patched_members

_START = 100.0
_INTERVAL = 0.1
# The first tick's work overruns to 3.5 intervals after the start: ticks 2 and 3 have passed.
_FIRST_TICK_ENDS = _START + 3.5 * _INTERVAL


class _Target:
    """A weakly referenceable event target the timer never reaches."""


class _RecordingTimer(Timer):
    def __init__(self, clock: list[float]) -> None:
        self.target_holder = _Target()
        super().__init__(self.target_holder, _INTERVAL)
        self.clock = clock
        self.ticks: list[float] = []

    async def _tick(self, *, next_timer: float, count: int) -> None:
        self.ticks.append(next_timer)
        if len(self.ticks) == 2:
            raise EventTargetGone()
        self.clock[0] = _FIRST_TICK_ENDS


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [_START]

    async def sleep(delay: float) -> None:
        clock[0] += delay

    monkeypatch.setattr(timer_module, "_time", SimpleNamespace(get_time=lambda: clock[0]))
    monkeypatch.setattr(timer_module, "sleep", sleep)
    monkeypatch.setattr(Timer, "_run", Timer._run)
    return clock


async def test_a_late_tick_resumes_on_the_first_tick_still_ahead(fake_clock: list[float]) -> None:
    textual_timer_skip.apply_runtime_patch()
    timer = _RecordingTimer(fake_clock)

    await timer._run()

    assert timer.ticks == [pytest.approx(_START + _INTERVAL), pytest.approx(_START + 4 * _INTERVAL)]


async def test_upstream_skips_one_tick_too_many(fake_clock: list[float]) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    upstream = FilePatch(
        package="textual",
        module_file="timer.py",
        old_fragment=textual_timer_skip._NEW,
        new_fragment=textual_timer_skip._OLD,
        description="Restore the upstream skip count",
    )
    assert install_patched_members(timer_module, [upstream], {"Timer": ["_run"]}, marker="_upstream", label="test")
    timer = _RecordingTimer(fake_clock)

    await timer._run()

    assert timer.ticks == [pytest.approx(_START + _INTERVAL), pytest.approx(_START + 5 * _INTERVAL)]


def test_file_patch_fragment_matches_installed_textual() -> None:
    assert timer_module.__file__ is not None
    source = Path(timer_module.__file__).read_text(encoding="utf-8")
    assert textual_timer_skip._OLD in source or textual_timer_skip._NEW in source

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for GC-freeze calibration arithmetic."""

from __future__ import annotations

import importlib

import pytest
from scripts.gc_freeze_calibration_math import (
    aggregate_dead_cyclic_fraction,
    dead_cyclic_fraction,
    minimum_deferred_diff_surfaces,
    validated_absorb_points,
)
from textual.widgets import Static

from chrys.app.tui.support.gc_freeze import GcFreezeBlockReason
from tests.support.paths import REPO_ROOT
from tests.support.waiting import wait_until


def _calibration_module(monkeypatch: pytest.MonkeyPatch):
    """Import the executable script with its sibling-import path available."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    return importlib.import_module("scripts.calibrate_gc_freeze")


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        (1, 1),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 9),
        (12, 22),
    ],
)
def test_minimum_deferred_diff_surfaces_tolerates_worker_tail(turns: int, expected: int) -> None:
    assert minimum_deferred_diff_surfaces(turns) == expected


def test_absorb_points_must_not_exceed_soft_reclaim_threshold() -> None:
    assert validated_absorb_points([12, 0, 4, 4], max_absorbs=12) == (0, 4, 12)
    with pytest.raises(ValueError, match=r"\[0, 12\]"):
        validated_absorb_points([0, 13], max_absorbs=12)


def test_dead_cyclic_fraction_uses_inclusive_freeze_delta() -> None:
    assert dead_cyclic_fraction(
        collected_objects=40,
        zero_collected=10,
        frozen_objects_added_by_absorbs=100,
    ) == pytest.approx(0.30)


def test_aggregate_dead_cyclic_fraction_uses_total_observed_cohort() -> None:
    assert aggregate_dead_cyclic_fraction(
        [
            (40, 10, 100),
            (35, 10, 100),
        ]
    ) == pytest.approx(0.275)


@pytest.mark.parametrize(
    ("collected_objects", "zero_collected", "frozen_objects_added", "expected"),
    [
        (10, 10, 100, 0.0),
        (10, 10, 0, None),
        (11, 10, 0, 1.0),
    ],
)
def test_dead_cyclic_fraction_handles_zero_denominator(
    collected_objects: int,
    zero_collected: int,
    frozen_objects_added: int,
    expected: float | None,
) -> None:
    assert (
        dead_cyclic_fraction(
            collected_objects=collected_objects,
            zero_collected=zero_collected,
            frozen_objects_added_by_absorbs=frozen_objects_added,
        )
        == expected
    )


@pytest.mark.gc_calibration
@pytest.mark.asyncio
async def test_action_latency_includes_deferred_full_layout_passes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration_module(monkeypatch)
    app, _updater = calibration._build_app(tmp_path, enabled=True)
    async with app.run_test(size=(120, 36)) as pilot:
        await calibration._wait_for_initial_freeze(app=app, pilot=pilot)
        await app.screen.mount(*(Static("") for _ in range(512)))
        await pilot.pause()

        measurement = await calibration._request_absorb(coordinator=app._gc_freeze, pilot=pilot)

        assert measurement.layout_passes >= 1
        assert measurement.layout_ms > 0
        assert measurement.duration_ms == pytest.approx(measurement.metrics.duration_ms + measurement.layout_ms)


@pytest.mark.gc_calibration
@pytest.mark.asyncio
async def test_action_wait_allows_watchdog_retry_after_transient_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration_module(monkeypatch)
    app, _updater = calibration._build_app(tmp_path, enabled=True)
    async with app.run_test(size=(120, 36)) as pilot:
        await calibration._wait_for_initial_freeze(app=app, pilot=pilot)
        original_freeze_block_reason = app.freeze_block_reason
        gate_observed = False

        def transient_block() -> GcFreezeBlockReason | None:
            nonlocal gate_observed
            if not gate_observed:
                gate_observed = True
                return GcFreezeBlockReason.SCROLL_GC_PAUSED
            return original_freeze_block_reason()

        monkeypatch.setattr(app, "freeze_block_reason", transient_block)

        measurement = await calibration._request_absorb(coordinator=app._gc_freeze, pilot=pilot)

        assert gate_observed is True
        assert measurement.metrics.action == "absorb"
        assert app._gc_freeze._absorb_pending is False


@pytest.mark.gc_calibration
@pytest.mark.asyncio
async def test_soft_limit_absorb_cannot_trigger_watchdog_reclaim_before_measured_full(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration_module(monkeypatch)
    app, _updater = calibration._build_app(tmp_path, enabled=True)
    async with app.run_test(size=(120, 36)) as pilot:
        await calibration._wait_for_initial_freeze(app=app, pilot=pilot)
        for _ in range(12):
            measurement = await calibration._request_absorb(coordinator=app._gc_freeze, pilot=pilot)
            assert measurement.metrics.action == "absorb"

        absorb_metrics = app._gc_freeze.last_action_metrics
        assert app._gc_freeze._idle_reclaim_pending is True
        assert app._calibration_action_blocked is True
        assert not await wait_until(
            lambda: app._gc_freeze.last_action_metrics is not absorb_metrics,
            timeout=1.2,
            pilot=pilot,
        )
        assert app._gc_freeze.last_action_metrics is absorb_metrics

        full = await calibration._request_full_reclaim(coordinator=app._gc_freeze, pilot=pilot)
        assert full.metrics.action == "full"


@pytest.mark.gc_calibration
@pytest.mark.asyncio
async def test_collapse_probe_reopens_gate_for_watchdog_idle_reclaim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration_module(monkeypatch)
    app, _updater = calibration._build_app(tmp_path, enabled=True)
    async with app.run_test(size=(120, 36)) as pilot:
        await calibration._wait_for_initial_freeze(app=app, pilot=pilot)
        _screen, panel, _terminal = await calibration._populate_whole_app(
            app=app,
            pilot=pilot,
            turns=1,
            include_collapse_probe=True,
        )
        await calibration._request_full_reclaim(coordinator=app._gc_freeze, pilot=pilot)
        assert app._calibration_action_blocked is True

        result = await calibration._measure_tool_group_collapse(app=app, pilot=pilot, panel=panel)

        assert result.descendant_alive_while_frozen is True
        assert result.descendant_reclaimed is True
        assert app._calibration_action_blocked is True

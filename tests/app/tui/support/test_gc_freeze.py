# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the event-driven TUI GC freeze coordinator."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest
from textual import _time as textual_time

from chrys.app.tui.support import gc_freeze
from chrys.app.tui.support.gc_freeze import (
    GcAbsorbReason,
    GcAbsorbRequested,
    GcFreezeBlockReason,
    GcFreezeCoordinator,
    GcReclaimReason,
    GcReclaimRequested,
)


@dataclass
class _Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _Host:
    callbacks: list[Callable[[], None]] = field(default_factory=list)
    post_results: list[bool] = field(default_factory=list)
    block: GcFreezeBlockReason | None = None
    gate_calls: int = 0
    prepare_calls: int = 0
    after_calls: int = 0
    abort_calls: int = 0
    prepare_failures_remaining: int = 0
    after_failures_remaining: int = 0

    def call_after_refresh(self, callback: Callable[[], None]) -> bool:
        accepted = self.post_results.pop(0) if self.post_results else True
        if accepted:
            self.callbacks.append(callback)
        return accepted

    def freeze_block_reason(self) -> GcFreezeBlockReason | None:
        self.gate_calls += 1
        return self.block

    def prepare_for_gc_freeze(self) -> None:
        self.prepare_calls += 1
        if self.prepare_failures_remaining:
            self.prepare_failures_remaining -= 1
            raise RuntimeError("prepare failed")

    def after_gc_freeze(self) -> None:
        self.after_calls += 1
        if self.after_failures_remaining:
            self.after_failures_remaining -= 1
            raise RuntimeError("after failed")

    def abort_gc_freeze(self) -> None:
        self.abort_calls += 1

    def run_next_refresh(self) -> None:
        callback = self.callbacks.pop(0)
        callback()

    def run_settle_barrier(self) -> None:
        self.run_next_refresh()
        self.run_next_refresh()


@dataclass
class _GcRecorder:
    operations: list[str] = field(default_factory=list)
    freeze_count: int = 0
    freeze_count_reads: int = 0

    def collect(self) -> int:
        self.operations.append("collect")
        return 0

    def freeze(self) -> None:
        self.operations.append("freeze")
        self.freeze_count = 1

    def unfreeze(self) -> None:
        self.operations.append("unfreeze")
        self.freeze_count = 0

    def get_freeze_count(self) -> int:
        self.freeze_count_reads += 1
        return self.freeze_count


@pytest.fixture
def gc_recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_GcRecorder]:
    recorder = _GcRecorder()
    monkeypatch.setattr(gc_freeze.gc, "collect", recorder.collect)
    monkeypatch.setattr(gc_freeze.gc, "freeze", recorder.freeze)
    monkeypatch.setattr(gc_freeze.gc, "unfreeze", recorder.unfreeze)
    monkeypatch.setattr(gc_freeze.gc, "get_freeze_count", recorder.get_freeze_count)
    monkeypatch.setattr(gc_freeze, "_enabled_owner", None)
    yield recorder
    owner = gc_freeze._enabled_owner
    if owner is not None:
        owner.close()
    assert gc_freeze._enabled_owner is None


@pytest.fixture
def coordinators(gc_recorder: _GcRecorder) -> Iterator[Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]]]:
    created: list[GcFreezeCoordinator] = []

    def _make(
        *,
        host: _Host | None = None,
        clock: _Clock | None = None,
        enabled: bool = True,
        measure_freeze_counts: bool = False,
    ) -> tuple[GcFreezeCoordinator, _Host, _Clock]:
        host = host or _Host()
        clock = clock or _Clock()
        coordinator = GcFreezeCoordinator(
            host,
            enabled=enabled,
            clock=clock,
            measure_freeze_counts=measure_freeze_counts,
        )
        created.append(coordinator)
        return coordinator, host, clock

    yield _make
    for coordinator in reversed(created):
        coordinator.close()


def _start_and_freeze(
    coordinator: GcFreezeCoordinator,
    host: _Host,
    recorder: _GcRecorder,
) -> None:
    coordinator.start()
    assert recorder.operations == []
    assert len(host.callbacks) == 1
    host.run_next_refresh()
    assert recorder.operations == []
    assert len(host.callbacks) == 1
    host.run_next_refresh()
    assert recorder.operations == ["collect", "freeze"]
    assert coordinator.frozen is True
    recorder.operations.clear()


def test_request_messages_keep_textual_source_time() -> None:
    absorb = GcAbsorbRequested(GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
    reclaim = GcReclaimRequested(GcReclaimReason.SESSION_READY, prompt=True)

    assert isinstance(absorb.time, float)
    assert isinstance(reclaim.time, float)
    assert absorb.reason is GcAbsorbReason.TURN_TERMINAL
    assert reclaim.reason is GcReclaimReason.SESSION_READY


def test_initial_freeze_crosses_two_refresh_barriers(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators()

    _start_and_freeze(coordinator, host, gc_recorder)

    assert host.prepare_calls == 1
    assert host.after_calls == 1


def test_prompt_reclaim_runs_unfreeze_collect_freeze(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    coordinator.request_reclaim(reason=GcReclaimReason.SESSION_RESTORED, prompt=True, requested_at=clock())
    host.run_settle_barrier()

    assert gc_recorder.operations == ["unfreeze", "collect", "freeze"]
    assert coordinator._absorbs_since_reclaim == 0


def test_absorb_never_unfreezes(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_settle_barrier()

    assert gc_recorder.operations == ["collect", "freeze"]
    assert coordinator._absorbs_since_reclaim == 1


def test_full_reclaim_wins_over_absorb_and_clears_both_intents(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    coordinator.request_absorb(
        reason=GcAbsorbReason.PROFILE_UI_UPDATED,
        terminal_boundary=True,
        requested_at=clock(),
    )
    coordinator.request_reclaim(reason=GcReclaimReason.SESSION_READY, prompt=True, requested_at=clock())
    assert len(host.callbacks) == 1
    host.run_settle_barrier()

    assert gc_recorder.operations == ["unfreeze", "collect", "freeze"]
    assert coordinator._absorb_pending is False
    assert coordinator._prompt_reclaim_pending is False
    assert coordinator._absorb_reasons == set()
    assert coordinator._prompt_reclaim_reasons == set()


def test_absorb_before_initial_freeze_is_inert(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    coordinator.start()

    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )

    assert coordinator._absorb_pending is False
    assert len(host.callbacks) == 1
    host.run_settle_barrier()
    assert gc_recorder.operations == ["collect", "freeze"]


def test_requests_coalesce_before_dispatch(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    for reason in GcAbsorbReason:
        coordinator.request_absorb(reason=reason, terminal_boundary=True, requested_at=clock())

    assert len(host.callbacks) == 1
    assert coordinator._absorb_reasons == set(GcAbsorbReason)
    host.run_settle_barrier()
    assert gc_recorder.operations == ["collect", "freeze"]


@pytest.mark.parametrize("failed_hop", [1, 2])
def test_failed_refresh_post_clears_scheduled_immediately(
    failed_hop: int,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    host = _Host(post_results=[failed_hop != 1, failed_hop != 2])
    coordinator, host, _clock = coordinators(host=host)

    coordinator.start()
    if failed_hop == 2:
        host.run_next_refresh()

    assert coordinator._scheduled is False
    assert host.callbacks == []
    assert gc_recorder.operations == []


@pytest.mark.parametrize("lost_hop", [1, 2])
def test_expired_refresh_chain_is_replaced_and_stale_callback_is_inert(
    lost_hop: int,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    coordinator.start()
    if lost_hop == 2:
        host.run_next_refresh()
    stale_callback = host.callbacks.pop(0)
    stale_generation = coordinator._schedule_generation

    clock.advance(gc_freeze._SCHEDULE_RECOVERY_SECONDS)
    coordinator.on_tick()

    assert coordinator._schedule_generation > stale_generation
    assert coordinator._refresh_recovery_count == 1
    assert len(host.callbacks) == 1
    stale_callback()
    assert coordinator._scheduled is True
    assert gc_recorder.operations == []
    host.run_settle_barrier()
    assert gc_recorder.operations == ["collect", "freeze"]


def test_refresh_recovery_is_bounded_and_waits_for_gate(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    coordinator.start()
    host.callbacks.clear()
    host.block = GcFreezeBlockReason.TOP_SCREEN
    clock.advance(gc_freeze._SCHEDULE_RECOVERY_SECONDS)

    for _ in range(100):
        coordinator.on_tick()

    assert coordinator._refresh_recovery_count == 0
    assert host.callbacks == []

    host.block = None
    coordinator.on_tick()
    assert coordinator._refresh_recovery_count == 1
    assert len(host.callbacks) == 1
    coordinator.on_tick()
    assert coordinator._refresh_recovery_count == 1
    assert len(host.callbacks) == 1
    assert gc_recorder.operations == []


def test_no_work_ticks_do_not_query_gate_or_run_gc(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    host.gate_calls = 0

    for _ in range(5_000):
        coordinator.on_tick()

    assert host.gate_calls == 0
    assert gc_recorder.operations == []


def test_blocked_dispatch_leaves_intent_pending_until_watchdog_retry(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_next_refresh()
    host.block = GcFreezeBlockReason.AGENT_RUNNING
    host.run_next_refresh()

    assert coordinator._absorb_pending is True
    assert coordinator._scheduled is False
    assert host.callbacks == []
    assert gc_recorder.operations == []

    coordinator.on_tick()
    assert host.callbacks == []
    host.block = None
    coordinator.on_tick()
    assert len(host.callbacks) == 1
    host.run_settle_barrier()
    assert gc_recorder.operations == ["collect", "freeze"]


def test_newer_input_demotes_terminal_privilege_but_keeps_intent(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators(clock=_Clock(10.0))
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=20.0,
    )
    coordinator.note_input(occurred_at=21.0)
    host.run_settle_barrier()

    assert coordinator._terminal_boundary_at is None
    assert coordinator._absorb_pending is True
    assert gc_recorder.operations == []


def test_older_terminal_delivered_after_newer_input_cannot_regain_privilege(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators(clock=_Clock(10.0))
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.note_input(occurred_at=30.0)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=20.0,
    )
    host.run_settle_barrier()

    assert coordinator._terminal_boundary_at is None
    assert coordinator._absorb_pending is True
    assert gc_recorder.operations == []


def test_older_input_delivered_after_newer_terminal_keeps_privilege(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators(clock=_Clock(10.0))
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=30.0,
    )
    coordinator.note_input(occurred_at=20.0)
    host.run_settle_barrier()

    assert gc_recorder.operations == ["collect", "freeze"]


def test_absorb_request_older_than_last_action_is_discarded(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    action_at = coordinator._last_absorbing_action_at

    coordinator.request_absorb(
        reason=GcAbsorbReason.STABLE_CONTENT_MOUNTED,
        terminal_boundary=True,
        requested_at=action_at - 1,
    )

    assert coordinator._absorb_pending is False
    assert host.callbacks == []


def test_reclaim_delayed_across_absorb_remains_live_but_not_across_full(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    delayed_at = clock()
    clock.advance(1.0)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_settle_barrier()
    gc_recorder.operations.clear()

    coordinator.request_reclaim(
        reason=GcReclaimReason.AGENT_REBUILT,
        prompt=True,
        requested_at=delayed_at,
    )
    assert coordinator._prompt_reclaim_pending is True
    host.run_settle_barrier()
    assert gc_recorder.operations == ["unfreeze", "collect", "freeze"]
    gc_recorder.operations.clear()

    coordinator.request_reclaim(
        reason=GcReclaimReason.AGENT_REBUILT,
        prompt=True,
        requested_at=delayed_at,
    )
    assert coordinator._prompt_reclaim_pending is False
    assert host.callbacks == []


def test_idle_reclaim_waits_for_reclaim_idle(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.request_reclaim(reason=GcReclaimReason.AGENT_REBUILT, prompt=False, requested_at=clock())

    coordinator.on_tick()
    assert host.callbacks == []
    clock.advance(gc_freeze._RECLAIM_INPUT_IDLE_SECONDS)
    coordinator.on_tick()
    host.run_settle_barrier()

    assert gc_recorder.operations == ["unfreeze", "collect", "freeze"]


def test_hard_cap_promotes_terminal_absorb_to_full_reclaim(
    monkeypatch: pytest.MonkeyPatch,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    monkeypatch.setattr(gc_freeze, "_RECLAIM_SOFT_ABSORBS", 2)
    monkeypatch.setattr(gc_freeze, "_RECLAIM_HARD_ABSORBS", 3)
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator._absorbs_since_reclaim = 2

    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_settle_barrier()

    assert gc_recorder.operations == ["unfreeze", "collect", "freeze"]
    assert coordinator._absorbs_since_reclaim == 0


def test_successful_absorb_at_soft_threshold_places_idle_periodic_reclaim(
    monkeypatch: pytest.MonkeyPatch,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    monkeypatch.setattr(gc_freeze, "_RECLAIM_SOFT_ABSORBS", 2)
    monkeypatch.setattr(gc_freeze, "_RECLAIM_HARD_ABSORBS", 3)
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator._absorbs_since_reclaim = 1

    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_settle_barrier()

    assert coordinator._absorbs_since_reclaim == 2
    assert coordinator._idle_reclaim_pending is True
    assert coordinator._idle_reclaim_reasons == {GcReclaimReason.PERIODIC}


def test_whole_app_calibrated_reclaim_cadence_is_pinned() -> None:
    assert gc_freeze._RECLAIM_SOFT_ABSORBS == 12
    assert gc_freeze._RECLAIM_HARD_ABSORBS == 16


def test_production_actions_skip_permanent_generation_count_walks(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    startup_reads = gc_recorder.freeze_count_reads

    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.run_settle_barrier()

    assert gc_recorder.freeze_count_reads == startup_reads
    assert coordinator.last_action_metrics is not None
    assert coordinator.last_action_metrics.freeze_count_before is None
    assert coordinator.last_action_metrics.freeze_count_after is None


def test_action_metrics_record_duration_counts_and_classified_reasons(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    duration_now = 0.0

    def _duration_clock() -> float:
        nonlocal duration_now
        duration_now += 0.010
        return duration_now

    monkeypatch.setattr(gc_freeze, "perf_counter", _duration_clock)
    coordinator, host, clock = coordinators(measure_freeze_counts=True)
    with caplog.at_level(logging.DEBUG, logger=gc_freeze.__name__):
        _start_and_freeze(coordinator, host, gc_recorder)
        caplog.clear()
        coordinator.request_absorb(
            reason=GcAbsorbReason.PROFILE_UI_UPDATED,
            terminal_boundary=True,
            requested_at=clock(),
        )
        coordinator.request_reclaim(
            reason=GcReclaimReason.AGENT_REBUILT,
            prompt=False,
            requested_at=clock(),
        )
        host.run_settle_barrier()

    records = [record for record in caplog.records if record.__dict__.get("event_name") == "chrys.gc.freeze.action"]
    assert len(records) == 1
    record = records[0]
    assert record.__dict__["gc_action"] == "absorb"
    assert record.__dict__["gc_absorb_reasons"] == (GcAbsorbReason.PROFILE_UI_UPDATED.value,)
    assert record.__dict__["gc_prompt_reclaim_reasons"] == ()
    assert record.__dict__["gc_idle_reclaim_reasons"] == (GcReclaimReason.AGENT_REBUILT.value,)
    assert record.__dict__["gc_terminal"] is True
    assert record.__dict__["gc_collected"] == 0
    assert record.__dict__["gc_duration_ms"] == pytest.approx(10.0)
    assert record.__dict__["gc_freeze_count_before"] == 1
    assert record.__dict__["gc_freeze_count_after"] == 1
    assert record.__dict__["gc_absorbs_since_reclaim"] == 1


def test_deferral_recovery_and_pending_warning_are_structured_and_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    monkeypatch.setattr(gc_freeze, "_PENDING_WARNING_SECONDS", 1.0)
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )
    host.callbacks.clear()
    host.block = GcFreezeBlockReason.POINTER_BUTTON_HELD
    clock.advance(gc_freeze._SCHEDULE_RECOVERY_SECONDS)

    with caplog.at_level(logging.DEBUG, logger=gc_freeze.__name__):
        coordinator.on_tick()
        coordinator.on_tick()
        host.block = None
        coordinator.on_tick()

    deferrals = [record for record in caplog.records if record.__dict__.get("event_name") == "chrys.gc.freeze.deferral"]
    pending = [record for record in caplog.records if record.__dict__.get("event_name") == "chrys.gc.freeze.pending"]
    recoveries = [
        record for record in caplog.records if record.__dict__.get("event_name") == "chrys.gc.freeze.recovery"
    ]
    assert len(deferrals) == 1
    assert deferrals[0].__dict__["gc_deferral_reason"] == GcFreezeBlockReason.POINTER_BUTTON_HELD.value
    assert deferrals[0].__dict__["gc_scheduled_hop"] == 1
    assert len(pending) == 1
    assert pending[0].__dict__["gc_pending_seconds"] == pytest.approx(gc_freeze._SCHEDULE_RECOVERY_SECONDS)
    assert len(recoveries) == 1
    assert recoveries[0].__dict__["gc_expired_hop"] == 1
    assert recoveries[0].__dict__["gc_refresh_recovery_count"] == 1


def test_slow_absorb_warning_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    duration_now = 0.0

    def _duration_clock() -> float:
        nonlocal duration_now
        duration_now += 0.010
        return duration_now

    monkeypatch.setattr(gc_freeze, "perf_counter", _duration_clock)
    monkeypatch.setattr(gc_freeze, "_ABSORB_WARNING_SECONDS", 0.001)
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    with caplog.at_level(logging.WARNING, logger=gc_freeze.__name__):
        for _ in range(2):
            coordinator.request_absorb(
                reason=GcAbsorbReason.TURN_TERMINAL,
                terminal_boundary=True,
                requested_at=clock(),
            )
            host.run_settle_barrier()

    # Filter by level as well as event: booting a real ChrysApp anywhere earlier
    # on this worker leaves the top-level "chrys" logger at DEBUG
    # (install_log_handler's whitelist escalation), so the DEBUG action=initial
    # record from _start_and_freeze would otherwise be captured and counted.
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.__dict__.get("event_name") == "chrys.gc.freeze.action"
    ]
    assert len(warnings) == 1
    assert warnings[0].__dict__["gc_action"] == "absorb"


def test_sub_threshold_absorb_latency_growth_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    duration_pairs = iter(
        [
            0.000,
            0.010,
            0.100,
            0.112,
            0.200,
            0.215,
            0.300,
            0.318,
            0.400,
            0.419,
        ]
    )
    monkeypatch.setattr(gc_freeze, "perf_counter", lambda: next(duration_pairs, 0.500))

    with caplog.at_level(logging.WARNING, logger=gc_freeze.__name__):
        for _ in range(5):
            coordinator.request_absorb(
                reason=GcAbsorbReason.TURN_TERMINAL,
                terminal_boundary=True,
                requested_at=clock(),
            )
            host.run_settle_barrier()

    trend_warnings = [
        record for record in caplog.records if record.__dict__.get("gc_latency_warning") == "upward_trend"
    ]
    assert len(trend_warnings) == 1
    assert trend_warnings[0].__dict__["gc_absorb_trend_durations_ms"] == pytest.approx((10, 12, 15, 18))
    assert not [
        record for record in caplog.records if record.__dict__.get("gc_latency_warning") == "absolute_threshold"
    ]
    assert len(coordinator._recent_absorb_durations) == gc_freeze._ABSORB_TREND_WINDOW
    coordinator.request_reclaim(reason=GcReclaimReason.SESSION_READY, prompt=True, requested_at=clock())
    host.run_settle_barrier()
    assert not coordinator._recent_absorb_durations


def test_prepare_failures_retry_after_one_then_two_seconds_and_fail_open(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)
    host.prepare_failures_remaining = 3
    coordinator.request_absorb(
        reason=GcAbsorbReason.TURN_TERMINAL,
        terminal_boundary=True,
        requested_at=clock(),
    )

    host.run_settle_barrier()
    assert gc_recorder.operations == []
    assert coordinator._prepare_retry_not_before == pytest.approx(clock() + 1.0)
    coordinator.on_tick()
    assert host.callbacks == []
    clock.advance(1.0)
    coordinator.on_tick()
    host.run_settle_barrier()
    assert gc_recorder.operations == []
    assert coordinator._prepare_retry_not_before == pytest.approx(clock() + 2.0)
    clock.advance(2.0)
    coordinator.on_tick()
    host.run_settle_barrier()

    assert gc_recorder.operations == ["unfreeze", "collect"]
    assert coordinator._faulted is True
    assert coordinator.frozen is False
    assert coordinator._absorb_pending is False
    assert host.abort_calls == 3


def test_prepare_failure_exhaustion_before_initial_freeze_runs_no_gc(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    host = _Host(prepare_failures_remaining=3)
    coordinator, host, clock = coordinators(host=host)
    coordinator.start()

    host.run_settle_barrier()
    clock.advance(1.0)
    coordinator.on_tick()
    host.run_settle_barrier()
    clock.advance(2.0)
    coordinator.on_tick()
    host.run_settle_barrier()

    assert gc_recorder.operations == []
    assert coordinator._faulted is True
    assert coordinator.frozen is False
    assert host.abort_calls == 3


def test_after_hook_failure_on_initial_freeze_restores_unfrozen_baseline(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    host = _Host(after_failures_remaining=1)
    coordinator, host, _clock = coordinators(host=host)
    coordinator.start()
    host.run_settle_barrier()

    assert gc_recorder.operations == ["collect", "freeze", "unfreeze", "collect"]
    assert gc_recorder.freeze_count == 0
    assert coordinator._faulted is True
    assert coordinator.frozen is False
    assert host.abort_calls == 1


def test_disabled_coordinator_is_fully_inert(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, clock = coordinators(enabled=False)
    last_input = coordinator._last_input_at

    coordinator.start()
    coordinator.note_input(occurred_at=clock() + 10)
    coordinator.request_reclaim(reason=GcReclaimReason.SESSION_READY, prompt=True)
    coordinator.request_absorb(reason=GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
    for _ in range(10):
        coordinator.on_tick()

    assert coordinator._last_input_at == last_input
    assert host.callbacks == []
    assert gc_recorder.operations == []


def test_production_constructor_uses_textual_clock() -> None:
    coordinator = GcFreezeCoordinator(_Host(), enabled=False)

    assert coordinator._clock is textual_time.get_time


def test_close_invalidates_posted_callback_and_only_unfreezes_when_frozen(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators()
    coordinator.start()
    stale_callback = host.callbacks.pop()
    generation = coordinator._schedule_generation

    coordinator.close()
    stale_callback()

    assert coordinator._schedule_generation == generation + 1
    assert gc_recorder.operations == []
    assert host.callbacks == []


def test_second_enabled_owner_is_rejected_and_close_releases_owner(
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
) -> None:
    first, _host, _clock = coordinators()
    second, _second_host, _second_clock = coordinators()
    first.start()

    with pytest.raises(RuntimeError, match="Only one enabled Chrys"):
        second.start()

    first.close()
    second.start()
    assert second.started is True


def test_foreign_freeze_baseline_warns_and_disables_without_unfreezing(
    caplog: pytest.LogCaptureFixture,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    gc_recorder.freeze_count = 17
    coordinator, host, _clock = coordinators()

    with caplog.at_level(logging.WARNING, logger=gc_freeze.__name__):
        coordinator.start()

    assert coordinator.enabled is False
    assert coordinator.started is False
    assert host.callbacks == []
    assert gc_recorder.operations == []
    assert coordinator.startup_warning == (
        "GC freeze optimization was disabled because this process already contains 17 externally frozen objects."
    )
    assert "already has 17 frozen objects" in caplog.text


def test_close_suppresses_cleanup_failure_and_releases_owner(
    monkeypatch: pytest.MonkeyPatch,
    coordinators: Callable[..., tuple[GcFreezeCoordinator, _Host, _Clock]],
    gc_recorder: _GcRecorder,
) -> None:
    coordinator, host, _clock = coordinators()
    _start_and_freeze(coordinator, host, gc_recorder)

    def _raise_unfreeze() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(gc_freeze.gc, "unfreeze", _raise_unfreeze)
    coordinator.close()
    monkeypatch.setattr(gc_freeze.gc, "unfreeze", gc_recorder.unfreeze)

    assert gc_freeze._enabled_owner is None
    assert coordinator.frozen is False

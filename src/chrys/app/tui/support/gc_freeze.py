# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Event-driven GC freeze coordination for the Textual UI."""

from __future__ import annotations

import gc
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import pairwise
from time import perf_counter
from typing import Any, Protocol, cast

from textual import _time as textual_time
from textual.cache import FIFOCache, LRUCache
from textual.message import Message
from textual.screen import Screen

logger = logging.getLogger(__name__)

GC_FREEZE_ENABLED = True
"""Whether a ``ChrysApp`` freezes the permanent generation by default.

Deliberately not a setting: the optimization is invisible when it works and
the failure modes it guards against are ours to fix, not a user's to
configure. ``ChrysApp(gc_freeze_enabled=...)`` overrides it for the
acceptance tests and the calibration harness, which is the whole audience.
"""

_WATCHDOG_PERIOD_SECONDS = 1.0
GC_FREEZE_WATCHDOG_PERIOD_SECONDS = _WATCHDOG_PERIOD_SECONDS
_SCHEDULE_RECOVERY_SECONDS = 3.0
_ABSORB_INPUT_IDLE_SECONDS = 0.75
_RECLAIM_INPUT_IDLE_SECONDS = 3.0
_RECLAIM_SOFT_ABSORBS = 12
_RECLAIM_HARD_ABSORBS = 16
_FROZEN_DEAD_BUDGET_BYTES = 32 * 1024 * 1024
_PREPARE_MAX_FAILURES = 3
_PREPARE_RETRY_BASE_SECONDS = 1.0
_DEFERRAL_LOG_INTERVAL_SECONDS = 30.0
_PENDING_WARNING_SECONDS = 30.0
_ABSORB_WARNING_SECONDS = 0.050
_FULL_RECLAIM_WARNING_SECONDS = 0.750
_ABSORB_TREND_WINDOW = 4
_ABSORB_TREND_GROWTH_RATIO = 1.5
_ABSORB_TREND_MIN_DELTA_SECONDS = 0.005


class GcAbsorbReason(Enum):
    """Stable UI mutation that should be added to the permanent generation."""

    TURN_TERMINAL = "turn_terminal"
    STABLE_CONTENT_MOUNTED = "stable_content_mounted"
    PROFILE_UI_UPDATED = "profile_ui_updated"
    WORKSPACE_UI_UPDATED = "workspace_ui_updated"
    VIEWPORT_UPDATED = "viewport_updated"


class GcReclaimReason(Enum):
    """Lifecycle boundary that may leave unreachable frozen cycles."""

    SESSION_READY = "session_ready"
    SESSION_RESTORED = "session_restored"
    ROLLBACK_WELCOME = "rollback_welcome"
    AGENT_REBUILT = "agent_rebuilt"
    AGENT_REBUILD_FAILED = "agent_rebuild_failed"
    STABLE_CONTENT_REMOVED = "stable_content_removed"
    THEME_UPDATED = "theme_updated"
    PERIODIC = "periodic"


class GcFreezeBlockReason(Enum):
    """Hard gate that makes a GC freeze action unsafe right now."""

    TOP_SCREEN = "top_screen"
    BACKEND_TURN_LIFECYCLE = "backend_turn_lifecycle"
    AGENT_LOADING = "agent_loading"
    AGENT_RUNNING = "agent_running"
    SCROLL_GC_PAUSED = "scroll_gc_paused"
    POINTER_BUTTON_HELD = "pointer_button_held"
    SESSION_JSON_VISIBLE = "session_json_visible"
    TRAJECTORY_DASHBOARD_VISIBLE = "trajectory_dashboard_visible"
    SHELL_VISIBLE = "shell_visible"
    SUGGESTIONS_VISIBLE = "suggestions_visible"
    PARTICIPANT = "participant"


@dataclass
class GcAbsorbRequested(Message):
    """Request an additive freeze after a stable UI mutation."""

    reason: GcAbsorbReason
    terminal_boundary: bool = False


@dataclass
class GcReclaimRequested(Message):
    """Request a full reclaim at a prompt or idle lifecycle boundary."""

    reason: GcReclaimReason
    prompt: bool


@dataclass(frozen=True, slots=True)
class GcFreezeActionMetrics:
    """Content-free measurements for one completed coordinator action."""

    action: str
    absorb_reasons: tuple[str, ...]
    prompt_reclaim_reasons: tuple[str, ...]
    idle_reclaim_reasons: tuple[str, ...]
    terminal: bool
    collected: int
    duration_ms: float
    freeze_count_before: int | None
    freeze_count_after: int | None
    absorbs_since_reclaim: int
    refresh_recovery_count: int


@dataclass(frozen=True, slots=True)
class DetachedLruCache:
    """Acyclic capacity token held only across a synchronous GC freeze action."""

    maxsize: int


@dataclass(frozen=True, slots=True)
class DetachedFifoCache:
    """Acyclic capacity token for a Textual arrangement FIFO."""

    maxsize: int


def detach_lru_cache(cache: LRUCache[Any, Any] | DetachedLruCache) -> DetachedLruCache:
    """Drop a cyclic LRU before collection without freezing a replacement head."""
    if isinstance(cache, DetachedLruCache):
        return cache
    return DetachedLruCache(maxsize=cache.maxsize)


def renew_lru_cache(detached: DetachedLruCache | LRUCache[Any, Any]) -> LRUCache[Any, Any]:
    """Create the mutable LRU only after the permanent generation changes."""
    if isinstance(detached, LRUCache):
        return detached
    return LRUCache(maxsize=detached.maxsize)


def detach_fifo_cache(cache: FIFOCache[Any, Any] | DetachedFifoCache) -> DetachedFifoCache:
    """Drop an arrangement FIFO before freezing its empty dictionary shell."""
    if isinstance(cache, DetachedFifoCache):
        return cache
    return DetachedFifoCache(maxsize=cache._maxsize)


def renew_fifo_cache(detached: DetachedFifoCache | FIFOCache[Any, Any]) -> FIFOCache[Any, Any]:
    """Recreate an arrangement FIFO after the permanent generation changes."""
    if isinstance(detached, FIFOCache):
        return detached
    return FIFOCache(maxsize=detached.maxsize)


def raise_gc_freeze_hook_errors(message: str, errors: list[Exception]) -> None:
    """Raise every independently observed hook failure as one diagnostic."""
    if errors:
        raise ExceptionGroup(message, errors)


class GcFreezeHost(Protocol):
    """Narrow App adapter used by :class:`GcFreezeCoordinator`."""

    def call_after_refresh(self, callback: Callable[[], None]) -> bool:
        """Schedule ``callback`` after the App's next refresh."""

    def freeze_block_reason(self) -> GcFreezeBlockReason | None:
        """Return the first active hard gate, if any."""

    def prepare_for_gc_freeze(self) -> None:
        """Prepare registered surfaces immediately before a GC action."""

    def after_gc_freeze(self) -> None:
        """Renew registered mutable cyclic surfaces after a freeze."""

    def abort_gc_freeze(self) -> None:
        """Best-effort restore every cache detached by an incomplete hook sequence."""


class GcFreezeParticipant(Protocol):
    """Contract for a permanently mounted mutable or cache-heavy UI surface."""

    def gc_freeze_block_reason(self) -> GcFreezeBlockReason | None:
        """Return the participant's current hard gate, if any."""

    def prepare_for_gc_freeze(self) -> None:
        """Release young cyclic cache state before collection and freeze."""

    def after_gc_freeze(self) -> None:
        """Replace mutable cyclic containers after the permanent generation changes."""

    def abort_gc_freeze(self) -> None:
        """Idempotently restore detached caches after any hook failure."""


def prepare_textual_screen_for_gc(screen: Screen) -> None:
    """Release screen/compositor and detach standard cyclic LRUs before freezing."""
    screen._compositor.clear()
    for widget in screen.walk_children(with_self=True):
        widget._box_model_cache = cast(Any, detach_lru_cache(widget._box_model_cache))
        widget._query_one_cache = cast(Any, detach_lru_cache(widget._query_one_cache))
        widget._arrangement_cache = cast(Any, detach_fifo_cache(widget._arrangement_cache))


def after_textual_screen_gc_freeze(screen: Screen) -> None:
    """Renew every standard cache, continuing past individual failures."""
    errors = _normalize_textual_screen_gc_caches(screen)
    raise_gc_freeze_hook_errors("Textual screen cache renewal failed", errors)


def abort_textual_screen_gc_freeze(screen: Screen) -> None:
    """Best-effort normalize every screen cache after an incomplete hook sequence."""
    for error in _normalize_textual_screen_gc_caches(screen):
        logger.error("Failed to restore a detached Textual cache", exc_info=error)


def _normalize_textual_screen_gc_caches(screen: Screen) -> list[Exception]:
    """Restore detached screen caches without stopping at the first failure."""
    errors: list[Exception] = []
    for widget in screen.walk_children(with_self=True):
        try:
            widget._box_model_cache = renew_lru_cache(widget._box_model_cache)
        except Exception as error:
            errors.append(error)
        try:
            widget._query_one_cache = renew_lru_cache(widget._query_one_cache)
        except Exception as error:
            errors.append(error)
        try:
            widget._arrangement_cache = renew_fifo_cache(widget._arrangement_cache)
        except Exception as error:
            errors.append(error)
    try:
        screen.refresh(layout=True)
    except Exception as error:
        errors.append(error)
    return errors


_enabled_owner: GcFreezeCoordinator | None = None


class GcFreezeCoordinator:
    """Place explicit GC freeze/reclaim intents at safe Textual boundaries."""

    def __init__(
        self,
        host: GcFreezeHost,
        *,
        enabled: bool,
        clock: Callable[[], float] | None = None,
        measure_freeze_counts: bool = False,
    ) -> None:
        if not 0 < _RECLAIM_SOFT_ABSORBS <= _RECLAIM_HARD_ABSORBS:
            msg = "GC freeze reclaim thresholds must be positive and ordered"
            raise ValueError(msg)

        self.enabled = enabled
        self._host = host
        self._clock = textual_time.get_time if clock is None else clock
        self._measure_freeze_counts = measure_freeze_counts
        self._started = False
        self._closed = False
        self._faulted = False
        self._startup_warning: str | None = None
        self._frozen = False
        self._scheduled = False
        self._schedule_generation = 0
        self._scheduled_hop = 0
        self._scheduled_at = 0.0

        self._absorb_pending = False
        self._terminal_boundary_at: float | None = None
        self._prompt_reclaim_pending = False
        self._idle_reclaim_pending = False

        self._absorb_reasons: set[GcAbsorbReason] = set()
        self._prompt_reclaim_reasons: set[GcReclaimReason] = set()
        self._idle_reclaim_reasons: set[GcReclaimReason] = set()

        self._absorbs_since_reclaim = 0
        self._last_input_at = self._clock()
        self._last_absorbing_action_at = float("-inf")
        self._last_full_action_at = float("-inf")
        self._prepare_failures = 0
        self._prepare_retry_not_before = float("-inf")

        self._last_deferral_reason: GcFreezeBlockReason | None = None
        self._last_deferral_logged_at = float("-inf")
        self._refresh_recovery_count = 0
        self._pending_since: float | None = None
        self._pending_warning_emitted = False
        self._absorb_warning_emitted = False
        self._absorb_trend_warning_emitted = False
        self._full_warning_emitted = False
        self._recent_absorb_durations: deque[float] = deque(maxlen=_ABSORB_TREND_WINDOW)
        self._last_action_metrics: GcFreezeActionMetrics | None = None

    @property
    def started(self) -> bool:
        """Whether this coordinator acquired the process-global lifecycle."""
        return self._started

    @property
    def frozen(self) -> bool:
        """Whether this coordinator currently owns frozen GC state."""
        return self._frozen

    @property
    def startup_warning(self) -> str | None:
        """User-facing reason the requested optimization degraded to disabled."""
        return self._startup_warning

    @property
    def last_action_metrics(self) -> GcFreezeActionMetrics | None:
        """Return the most recent completed action measurement, if any."""
        return self._last_action_metrics

    def start(self) -> None:
        """Acquire process ownership and queue the initial freeze."""
        global _enabled_owner

        if not self.enabled or self._closed or self._faulted or self._started:
            return
        if _enabled_owner is not None:
            msg = "Only one enabled Chrys GC freeze coordinator may run in a process"
            raise RuntimeError(msg)
        foreign_freeze_count = gc.get_freeze_count()
        if foreign_freeze_count:
            self.enabled = False
            self._startup_warning = (
                "GC freeze optimization was disabled because this process already contains "
                f"{foreign_freeze_count:,} externally frozen objects."
            )
            logger.warning(
                "Chrys GC freeze optimization disabled because the process already has %d frozen objects",
                foreign_freeze_count,
            )
            return

        _enabled_owner = self
        self._started = True
        self._mark_pending()
        self._schedule()

    def request_absorb(
        self,
        *,
        reason: GcAbsorbReason,
        terminal_boundary: bool,
        requested_at: float | None = None,
    ) -> None:
        """Coalesce an additive freeze request and schedule its settle barrier."""
        if self._inert() or not self._frozen:
            return
        source_time = self._clock() if requested_at is None else requested_at
        if source_time < max(self._last_absorbing_action_at, self._last_full_action_at):
            return

        self._absorb_pending = True
        self._absorb_reasons.add(reason)
        self._mark_pending()
        if terminal_boundary and source_time >= self._last_input_at:
            if self._terminal_boundary_at is None:
                self._terminal_boundary_at = source_time
            else:
                self._terminal_boundary_at = max(self._terminal_boundary_at, source_time)
        self._schedule()

    def request_reclaim(
        self,
        *,
        reason: GcReclaimReason,
        prompt: bool,
        requested_at: float | None = None,
    ) -> None:
        """Coalesce a full-reclaim request."""
        if self._inert():
            return
        source_time = self._clock() if requested_at is None else requested_at
        if source_time < self._last_full_action_at:
            return

        if prompt:
            self._prompt_reclaim_pending = True
            self._prompt_reclaim_reasons.add(reason)
            self._mark_pending()
            self._schedule()
        else:
            self._idle_reclaim_pending = True
            self._idle_reclaim_reasons.add(reason)
            self._mark_pending()

    def note_input(self, *, occurred_at: float) -> None:
        """Record source-time activity and demote older terminal privilege."""
        if self._inert():
            return
        self._last_input_at = max(self._last_input_at, occurred_at)
        if self._terminal_boundary_at is not None and occurred_at > self._terminal_boundary_at:
            self._terminal_boundary_at = None

    def on_tick(self) -> None:
        """Retry pending work, idle reclaim, or an expired refresh chain once."""
        if self._inert():
            return
        if self._scheduled:
            if self._clock() - self._scheduled_at < _SCHEDULE_RECOVERY_SECONDS:
                self._warn_if_pending_too_long()
                return
            if (block := self._host.freeze_block_reason()) is not None:
                self._record_deferral(block)
                return
            expired_generation = self._schedule_generation
            expired_hop = self._scheduled_hop
            self._schedule_generation += 1
            self._scheduled = False
            self._scheduled_hop = 0
            self._refresh_recovery_count += 1
            self._record_recovery(expired_generation=expired_generation, expired_hop=expired_hop)
            self._schedule()
            return
        if not (not self._frozen or self._prompt_reclaim_pending or self._idle_reclaim_pending or self._absorb_pending):
            return
        if self._clock() < self._prepare_retry_not_before:
            return
        if (block := self._host.freeze_block_reason()) is not None:
            self._record_deferral(block)
            return
        self._warn_if_pending_too_long()
        if (
            not self._frozen
            or self._prompt_reclaim_pending
            or (self._idle_reclaim_pending and self._reclaim_input_idle())
            or (
                self._absorb_pending
                and (
                    self._terminal_privileged()
                    or (self._absorbs_since_reclaim < _RECLAIM_HARD_ABSORBS and self._absorb_input_idle())
                )
            )
        ):
            self._schedule()

    def close(self) -> None:
        """Make callbacks inert, restore an unfrozen baseline, and release ownership."""
        global _enabled_owner

        was_started = self._started and not self._closed
        reason_snapshot = self._reason_snapshot()
        terminal = self._terminal_privileged()
        freeze_count_before = self._action_freeze_count()
        started_at = perf_counter()
        collected = 0
        self._closed = True
        self._schedule_generation += 1
        self._scheduled = False
        self._scheduled_hop = 0
        self._clear_all_intents()
        try:
            if self._frozen:
                gc.unfreeze()
                collected = gc.collect()
        except Exception:
            logger.exception("Failed to return GC to an unfrozen baseline during coordinator close")
        finally:
            self._frozen = False
            if _enabled_owner is self:
                _enabled_owner = None
        if was_started:
            self._record_action(
                "close",
                collected=collected,
                started_at=started_at,
                freeze_count_before=freeze_count_before,
                terminal=terminal,
                reason_snapshot=reason_snapshot,
            )

    def _inert(self) -> bool:
        return not self.enabled or not self._started or self._closed or self._faulted

    def _schedule(self) -> None:
        if self._inert() or self._scheduled:
            return
        if self._clock() < self._prepare_retry_not_before:
            return
        self._schedule_generation += 1
        generation = self._schedule_generation
        self._scheduled = True
        self._scheduled_hop = 1
        self._scheduled_at = self._clock()
        callback = partial(self._after_first_refresh, generation)
        if not self._host.call_after_refresh(callback):
            self._clear_schedule(generation)

    def _clear_schedule(self, generation: int) -> None:
        if generation == self._schedule_generation:
            self._scheduled = False
            self._scheduled_hop = 0

    def _after_first_refresh(self, generation: int) -> None:
        if generation != self._schedule_generation:
            return
        if self._inert():
            self._clear_schedule(generation)
            return
        self._scheduled_hop = 2
        self._scheduled_at = self._clock()
        callback = partial(self._run, generation)
        if not self._host.call_after_refresh(callback):
            self._clear_schedule(generation)

    def _run(self, generation: int) -> None:
        if generation != self._schedule_generation:
            return
        self._clear_schedule(generation)
        if self._inert():
            return
        if (block := self._host.freeze_block_reason()) is not None:
            self._record_deferral(block)
            return

        terminal = self._terminal_privileged()
        reclaim_idle = self._reclaim_input_idle()
        absorb_idle = self._absorb_input_idle()
        hard_reclaim_waiting = self._absorbs_since_reclaim >= _RECLAIM_HARD_ABSORBS
        full_due = (
            self._prompt_reclaim_pending
            or (self._idle_reclaim_pending and reclaim_idle)
            or (hard_reclaim_waiting and (reclaim_idle or terminal))
            or (terminal and self._absorb_pending and self._absorbs_since_reclaim + 1 >= _RECLAIM_HARD_ABSORBS)
        )

        if not self._frozen:
            self._run_initial_freeze()
            return
        if full_due:
            self._run_full_reclaim()
            return
        if self._absorb_pending and not hard_reclaim_waiting and (terminal or absorb_idle):
            self._run_absorb()

    def _run_initial_freeze(self) -> None:
        reason_snapshot = self._reason_snapshot()
        terminal = self._terminal_privileged()
        freeze_count_before = self._action_freeze_count()
        started_at = perf_counter()
        if not self._prepare_for_freeze():
            return
        collected = gc.collect()
        gc.freeze()
        if not self._finish_freeze():
            return
        self._frozen = True
        action_at = self._clock()
        self._last_absorbing_action_at = action_at
        self._last_full_action_at = action_at
        self._absorbs_since_reclaim = 0
        self._recent_absorb_durations.clear()
        self._record_action(
            "initial",
            collected=collected,
            started_at=started_at,
            freeze_count_before=freeze_count_before,
            terminal=terminal,
            reason_snapshot=reason_snapshot,
        )
        self._clear_all_intents()

    def _run_full_reclaim(self) -> None:
        reason_snapshot = self._reason_snapshot()
        terminal = self._terminal_privileged()
        freeze_count_before = self._action_freeze_count()
        started_at = perf_counter()
        if not self._prepare_for_freeze():
            return
        gc.unfreeze()
        collected = gc.collect()
        gc.freeze()
        if not self._finish_freeze():
            return
        self._frozen = True
        action_at = self._clock()
        self._last_absorbing_action_at = action_at
        self._last_full_action_at = action_at
        self._absorbs_since_reclaim = 0
        self._recent_absorb_durations.clear()
        self._record_action(
            "full",
            collected=collected,
            started_at=started_at,
            freeze_count_before=freeze_count_before,
            terminal=terminal,
            reason_snapshot=reason_snapshot,
        )
        self._clear_all_intents()

    def _run_absorb(self) -> None:
        reason_snapshot = self._reason_snapshot()
        terminal = self._terminal_privileged()
        freeze_count_before = self._action_freeze_count()
        started_at = perf_counter()
        idle_reclaim_was_pending = self._idle_reclaim_pending
        if not self._prepare_for_freeze():
            return
        collected = gc.collect()
        gc.freeze()
        if not self._finish_freeze():
            return
        action_at = self._clock()
        self._last_absorbing_action_at = action_at
        self._absorb_pending = False
        self._terminal_boundary_at = None
        self._absorb_reasons.clear()
        self._absorbs_since_reclaim += 1
        if not idle_reclaim_was_pending:
            self._reset_pending_diagnostics()
        if self._absorbs_since_reclaim >= _RECLAIM_SOFT_ABSORBS:
            self._idle_reclaim_pending = True
            self._idle_reclaim_reasons.add(GcReclaimReason.PERIODIC)
            self._mark_pending(at=action_at)
        self._record_action(
            "absorb",
            collected=collected,
            started_at=started_at,
            freeze_count_before=freeze_count_before,
            terminal=terminal,
            reason_snapshot=reason_snapshot,
        )

    def _prepare_for_freeze(self) -> bool:
        try:
            self._host.prepare_for_gc_freeze()
        except Exception:
            self._abort_gc_freeze_hooks()
            self._prepare_failures += 1
            logger.exception(
                "GC freeze participant preparation failed (attempt %d/%d)",
                self._prepare_failures,
                _PREPARE_MAX_FAILURES,
            )
            if self._prepare_failures >= _PREPARE_MAX_FAILURES:
                self._fail_open(permanent_generation_changed=False)
            else:
                delay = _PREPARE_RETRY_BASE_SECONDS * (2 ** (self._prepare_failures - 1))
                self._prepare_retry_not_before = self._clock() + delay
            return False
        self._prepare_failures = 0
        self._prepare_retry_not_before = float("-inf")
        return True

    def _finish_freeze(self) -> bool:
        try:
            self._host.after_gc_freeze()
        except Exception:
            self._abort_gc_freeze_hooks()
            logger.exception("GC freeze participant after-hook failed")
            self._fail_open(permanent_generation_changed=True)
            return False
        return True

    def _abort_gc_freeze_hooks(self) -> None:
        """Best-effort normalize live UI caches after an incomplete hook pass."""
        try:
            self._host.abort_gc_freeze()
        except Exception:
            logger.exception("GC freeze hook rollback failed")

    def _fail_open(self, *, permanent_generation_changed: bool) -> None:
        reason_snapshot = self._reason_snapshot()
        terminal = self._terminal_privileged()
        freeze_count_before = self._action_freeze_count()
        started_at = perf_counter()
        collected = 0
        should_unfreeze = permanent_generation_changed or self._frozen
        try:
            if should_unfreeze:
                gc.unfreeze()
                collected = gc.collect()
        except Exception:
            logger.exception("Failed to restore GC baseline while faulting freeze optimization")
        finally:
            self._frozen = False
            self._faulted = True
            self._clear_all_intents()
        self._record_action(
            "fail-open",
            collected=collected,
            started_at=started_at,
            freeze_count_before=freeze_count_before,
            terminal=terminal,
            reason_snapshot=reason_snapshot,
        )

    def _clear_all_intents(self) -> None:
        self._absorb_pending = False
        self._terminal_boundary_at = None
        self._prompt_reclaim_pending = False
        self._idle_reclaim_pending = False
        self._absorb_reasons.clear()
        self._prompt_reclaim_reasons.clear()
        self._idle_reclaim_reasons.clear()
        self._reset_pending_diagnostics()

    def _action_freeze_count(self) -> int | None:
        """Return a permanent-generation count only for offline calibration."""
        return gc.get_freeze_count() if self._measure_freeze_counts else None

    def _terminal_privileged(self) -> bool:
        return self._terminal_boundary_at is not None and self._terminal_boundary_at >= self._last_input_at

    def _absorb_input_idle(self) -> bool:
        return self._clock() - self._last_input_at >= _ABSORB_INPUT_IDLE_SECONDS

    def _reclaim_input_idle(self) -> bool:
        return self._clock() - self._last_input_at >= _RECLAIM_INPUT_IDLE_SECONDS

    def _record_deferral(self, reason: GcFreezeBlockReason) -> None:
        now = self._clock()
        self._warn_if_pending_too_long(now=now)
        if (
            reason is self._last_deferral_reason
            and now - self._last_deferral_logged_at < _DEFERRAL_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_deferral_reason = reason
        self._last_deferral_logged_at = now
        logger.debug(
            "GC freeze action deferred: %s",
            reason.value,
            extra={
                "event_name": "chrys.gc.freeze.deferral",
                "gc_deferral_reason": reason.value,
                "gc_pending_seconds": self._pending_seconds(now),
                "gc_absorb_pending": self._absorb_pending,
                "gc_prompt_reclaim_pending": self._prompt_reclaim_pending,
                "gc_idle_reclaim_pending": self._idle_reclaim_pending,
                "gc_schedule_generation": self._schedule_generation,
                "gc_scheduled_hop": self._scheduled_hop,
                "gc_refresh_recovery_count": self._refresh_recovery_count,
            },
        )

    def _record_recovery(self, *, expired_generation: int, expired_hop: int) -> None:
        logger.debug(
            "GC freeze refresh chain recovered: generation=%d hop=%d count=%d",
            expired_generation,
            expired_hop,
            self._refresh_recovery_count,
            extra={
                "event_name": "chrys.gc.freeze.recovery",
                "gc_expired_generation": expired_generation,
                "gc_expired_hop": expired_hop,
                "gc_refresh_recovery_count": self._refresh_recovery_count,
            },
        )

    def _record_action(
        self,
        action: str,
        *,
        collected: int,
        started_at: float,
        freeze_count_before: int | None,
        terminal: bool,
        reason_snapshot: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ) -> None:
        duration_seconds = max(0.0, perf_counter() - started_at)
        absorb_reasons, prompt_reclaim_reasons, idle_reclaim_reasons = reason_snapshot
        metrics = GcFreezeActionMetrics(
            action=action,
            absorb_reasons=absorb_reasons,
            prompt_reclaim_reasons=prompt_reclaim_reasons,
            idle_reclaim_reasons=idle_reclaim_reasons,
            terminal=terminal,
            collected=collected,
            duration_ms=duration_seconds * 1_000,
            freeze_count_before=freeze_count_before,
            freeze_count_after=self._action_freeze_count(),
            absorbs_since_reclaim=self._absorbs_since_reclaim,
            refresh_recovery_count=self._refresh_recovery_count,
        )
        self._last_action_metrics = metrics
        self._log_action(action)

    def _log_action(self, action: str) -> None:
        """Emit structured diagnostics for the most recently recorded action."""
        metrics = self._last_action_metrics
        if metrics is None or metrics.action != action:
            return
        extra: dict[str, object] = {
            "event_name": "chrys.gc.freeze.action",
            "gc_action": metrics.action,
            "gc_absorb_reasons": metrics.absorb_reasons,
            "gc_prompt_reclaim_reasons": metrics.prompt_reclaim_reasons,
            "gc_idle_reclaim_reasons": metrics.idle_reclaim_reasons,
            "gc_terminal": metrics.terminal,
            "gc_collected": metrics.collected,
            "gc_duration_ms": metrics.duration_ms,
            "gc_freeze_count_before": metrics.freeze_count_before,
            "gc_freeze_count_after": metrics.freeze_count_after,
            "gc_absorbs_since_reclaim": metrics.absorbs_since_reclaim,
            "gc_refresh_recovery_count": metrics.refresh_recovery_count,
        }
        logger.debug(
            "GC freeze action=%s duration_ms=%.3f collected=%d freeze_count=%s->%s "
            "absorb_reasons=%s prompt_reclaim_reasons=%s idle_reclaim_reasons=%s "
            "absorbs_since_reclaim=%d",
            action,
            extra["gc_duration_ms"],
            metrics.collected,
            metrics.freeze_count_before,
            extra["gc_freeze_count_after"],
            metrics.absorb_reasons,
            metrics.prompt_reclaim_reasons,
            metrics.idle_reclaim_reasons,
            metrics.absorbs_since_reclaim,
            extra=extra,
        )
        duration_seconds = metrics.duration_ms / 1_000
        if action == "absorb":
            self._record_absorb_latency(duration_seconds, extra=extra)
        elif (
            action in {"initial", "full"}
            and duration_seconds > _FULL_RECLAIM_WARNING_SECONDS
            and not self._full_warning_emitted
        ):
            self._full_warning_emitted = True
            logger.warning("GC freeze full action exceeded the calibrated latency threshold", extra=extra)

    def _record_absorb_latency(self, duration_seconds: float, *, extra: dict[str, object]) -> None:
        """Warn once for an absolute overrun and once for sustained sub-threshold growth."""
        self._recent_absorb_durations.append(duration_seconds)
        if duration_seconds > _ABSORB_WARNING_SECONDS and not self._absorb_warning_emitted:
            self._absorb_warning_emitted = True
            logger.warning(
                "GC freeze absorb exceeded the calibrated latency threshold",
                extra={**extra, "gc_latency_warning": "absolute_threshold"},
            )
        if self._absorb_trend_warning_emitted or len(self._recent_absorb_durations) < _ABSORB_TREND_WINDOW:
            return
        durations = tuple(self._recent_absorb_durations)
        if not all(earlier < later for earlier, later in pairwise(durations)):
            return
        if durations[-1] - durations[0] < _ABSORB_TREND_MIN_DELTA_SECONDS:
            return
        if durations[-1] < durations[0] * _ABSORB_TREND_GROWTH_RATIO:
            return
        self._absorb_trend_warning_emitted = True
        logger.warning(
            "GC freeze absorb latency is trending upward",
            extra={
                **extra,
                "gc_latency_warning": "upward_trend",
                "gc_absorb_trend_durations_ms": tuple(duration * 1_000 for duration in durations),
            },
        )

    def _reason_snapshot(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(sorted(reason.value for reason in self._absorb_reasons)),
            tuple(sorted(reason.value for reason in self._prompt_reclaim_reasons)),
            tuple(sorted(reason.value for reason in self._idle_reclaim_reasons)),
        )

    def _mark_pending(self, *, at: float | None = None) -> None:
        if self._pending_since is None:
            self._pending_since = self._clock() if at is None else at
            self._pending_warning_emitted = False

    def _reset_pending_diagnostics(self) -> None:
        self._pending_since = None
        self._pending_warning_emitted = False

    def _pending_seconds(self, now: float) -> float:
        return 0.0 if self._pending_since is None else max(0.0, now - self._pending_since)

    def _warn_if_pending_too_long(self, *, now: float | None = None) -> None:
        if self._pending_since is None or self._pending_warning_emitted:
            return
        current = self._clock() if now is None else now
        pending_seconds = self._pending_seconds(current)
        if pending_seconds < _PENDING_WARNING_SECONDS:
            return
        self._pending_warning_emitted = True
        logger.warning(
            "GC freeze request remains pending after %.1f seconds",
            pending_seconds,
            extra={
                "event_name": "chrys.gc.freeze.pending",
                "gc_pending_seconds": pending_seconds,
                "gc_absorb_pending": self._absorb_pending,
                "gc_prompt_reclaim_pending": self._prompt_reclaim_pending,
                "gc_idle_reclaim_pending": self._idle_reclaim_pending,
                "gc_schedule_generation": self._schedule_generation,
                "gc_scheduled_hop": self._scheduled_hop,
                "gc_refresh_recovery_count": self._refresh_recovery_count,
            },
        )


__all__ = [
    "GC_FREEZE_ENABLED",
    "GC_FREEZE_WATCHDOG_PERIOD_SECONDS",
    "DetachedFifoCache",
    "DetachedLruCache",
    "GcAbsorbReason",
    "GcAbsorbRequested",
    "GcFreezeActionMetrics",
    "GcFreezeBlockReason",
    "GcFreezeCoordinator",
    "GcFreezeHost",
    "GcFreezeParticipant",
    "GcReclaimReason",
    "GcReclaimRequested",
    "abort_textual_screen_gc_freeze",
    "after_textual_screen_gc_freeze",
    "detach_fifo_cache",
    "detach_lru_cache",
    "prepare_textual_screen_for_gc",
    "raise_gc_freeze_hook_errors",
    "renew_fifo_cache",
    "renew_lru_cache",
]

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Where a slow test's time went: the loop thread, the default executor and the process, sampled from a thread.

A CI runner stalls in ways the failing wait cannot explain: a ``pilot.pause`` that
spends thirty seconds on one widget, a five-second wait that misses by one. With
``CHRYS_TEST_STALL_DIAGNOSTICS`` set, a daemon thread samples every 0.2 s and, for
a test whose call phase ran long or failed, writes a report to the real stderr (a
duplicate of fd 2 taken before capture starts, as faulthandler does) so it lands
in the CI log next to the test.

The report says, for the loop thread, how much of the phase it sat in the
selector and what it was running when it did not; for the default executor, how
deep its queue was and which calls held its threads (on Windows every Textual
timer sleep is one of them, so its ``cpu_count + 4`` threads are the scarce
resource); and whether the sampler itself was ever kept from running, which is
the process or its VM being frozen rather than anything inside it.
``describe_pending_pumps`` names the widgets a timed-out Pilot wait still held.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import thread as futures_thread
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import FrameType

    from textual.message_pump import MessagePump

ENV_FLAG = "CHRYS_TEST_STALL_DIAGNOSTICS"
SAMPLE_INTERVAL_SECONDS = 0.2
REPORT_AFTER_SECONDS = 25.0
"""A green call phase longer than this is reported too: an outlier is the stall that stayed under the timeout."""
FREEZE_GAP_SECONDS = 2.0
"""The sampler oversleeping by this much means every thread stopped, not one of them."""
GC_PAUSE_SECONDS = 0.1
"""A collection at least this long is listed: it holds the GIL, so it stops every thread, the sampler included."""
_EXECUTOR_THREAD_PREFIX = "asyncio_"  # asyncio names its default executor's threads so
_LOOP_STACK_DEPTH = 6
_GC_PAUSES_KEPT = 256
_monotonic = time.monotonic
"""Bound at import: tests patch ``time.monotonic`` to a constant, and the phase clock must not follow them."""


def enabled() -> bool:
    return bool(os.environ.get(ENV_FLAG))


@dataclass
class PhaseRecord:
    """One test's call phase as the sampler saw it."""

    nodeid: str
    started_at: float
    interval: float
    ended_at: float | None = None
    samples: int = 0
    idle_samples: int = 0
    busy: Counter[str] = field(default_factory=Counter)
    longest_busy: tuple[str, float] = ("", 0.0)
    """The longest run of consecutive samples with one signature, as (signature, measured seconds)."""
    current_busy: tuple[str, float] = ("", 0.0)
    """The run in progress, as (signature, when its first sample was taken)."""
    executor_threads_max: int = 0
    executor_queue_max: int = 0
    executor_queue_total: int = 0
    executor_queued_samples: int = 0
    executor_busy: Counter[str] = field(default_factory=Counter)
    freezes: list[tuple[float, float, str]] = field(default_factory=list)
    """(offset into the phase, gap, what the loop thread was doing at the sample after the gap)."""
    gc_pauses: list[tuple[float, int, float]] = field(default_factory=list)
    """(offset into the phase, generation, seconds) of every collection at least ``GC_PAUSE_SECONDS`` long."""

    @property
    def seconds(self) -> float:
        return (self.ended_at if self.ended_at is not None else _monotonic()) - self.started_at


def _where(frame: FrameType) -> str:
    code = frame.f_code
    parts = code.co_filename.replace("\\", "/").rsplit("/", 2)
    return f"{code.co_name} ({'/'.join(parts[-2:])}:{frame.f_lineno})"


def _frames_outermost_first(frame: FrameType) -> list[FrameType]:
    frames: list[FrameType] = []
    current: FrameType | None = frame
    while current is not None:
        frames.append(current)
        current = current.f_back
    frames.reverse()
    return frames


def _in_selector(frame: FrameType) -> bool:
    """The loop thread is idle: asyncio waits for I/O in ``selectors.select`` or ``IocpProactor._poll``."""
    code = frame.f_code
    filename = code.co_filename.replace("\\", "/")
    return (code.co_name == "select" and filename.endswith("/selectors.py")) or (
        code.co_name == "_poll" and filename.endswith("/asyncio/windows_events.py")
    )


def _loop_signature(frame: FrameType) -> str:
    frames = _frames_outermost_first(frame)[-_LOOP_STACK_DEPTH:]
    return " <- ".join(_where(item) for item in reversed(frames))


def _executor_signature(frame: FrameType) -> str | None:
    """What an executor thread is running: the submitted call, then the innermost frame if it went deeper."""
    frames = _frames_outermost_first(frame)
    submitted = None
    for index, item in enumerate(frames[:-1]):
        code = item.f_code
        if code.co_name == "run" and code.co_filename.replace("\\", "/").endswith("concurrent/futures/thread.py"):
            submitted = frames[index + 1]
    if submitted is None:
        return None  # waiting on the work queue in ``_worker``
    signature = _where(submitted)
    leaf = frames[-1]
    return signature if leaf is submitted else f"{signature} ... {_where(leaf)}"


class StallSampler(threading.Thread):
    """Samples the loop thread and the default executor while a test's call phase is open."""

    def __init__(
        self,
        *,
        loop_thread_ident: int,
        sink: Callable[[str], None],
        interval: float = SAMPLE_INTERVAL_SECONDS,
        report_after: float = REPORT_AFTER_SECONDS,
        freeze_gap: float = FREEZE_GAP_SECONDS,
        gc_pause: float = GC_PAUSE_SECONDS,
    ) -> None:
        super().__init__(name="chrys-stall-sampler", daemon=True)
        self._loop_thread_ident = loop_thread_ident
        self._sink = sink
        self._interval = interval
        self._report_after = report_after
        self._freeze_gap = freeze_gap
        self._gc_pause = gc_pause
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase: PhaseRecord | None = None
        # Written by the gc callback on whichever thread collects, without the lock: an append is atomic.
        self._gc_started: float | None = None
        self._gc_pauses: list[tuple[float, int, float]] = []

    def start(self) -> None:
        gc.callbacks.append(self._on_gc)
        super().start()

    def begin_phase(self, nodeid: str) -> PhaseRecord:
        """Open a phase; the sampler thread fills in the returned record until :meth:`end_phase`."""
        with self._lock:
            self._phase = phase = PhaseRecord(nodeid, _monotonic(), self._interval)
        return phase

    def sampled(self) -> tuple[int, int]:
        """(samples, idle samples) of the open phase so far; a lock-free read for tests that wait on the count."""
        phase = self._phase
        return (phase.samples, phase.idle_samples) if phase is not None else (0, 0)

    def end_phase(self, *, failed: bool) -> PhaseRecord | None:
        """Close the phase; a failed or long one is written to the sink. Returns the record either way."""
        with self._lock:
            phase, self._phase = self._phase, None
        if phase is None:
            return None
        phase.ended_at = _monotonic()
        phase.gc_pauses = [
            (at - phase.started_at, generation, seconds)
            for at, generation, seconds in list(self._gc_pauses)
            if phase.started_at <= at <= phase.ended_at
        ]
        if failed or phase.seconds >= self._report_after:
            self._sink(format_report(phase, failed=failed))
        return phase

    def close(self) -> None:
        self._stop.set()
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if self.is_alive():
            self.join(timeout=max(1.0, self._interval * 5))

    def run(self) -> None:
        last = _monotonic()
        while not self._stop.wait(self._interval):
            last = self._sample(last)

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            self._gc_started = _monotonic()
            return
        started = self._gc_started
        if started is None:
            return
        seconds = _monotonic() - started
        if seconds >= self._gc_pause:
            self._gc_pauses.append((started, info["generation"], seconds))
            del self._gc_pauses[:-_GC_PAUSES_KEPT]

    def _sample(self, last: float) -> float:
        """Take one sample and return its time, read under the lock the phase opens and closes under.

        A time read before the lock could predate the phase the sample joins.
        """
        with self._lock:
            now = _monotonic()
            gap = now - last
            phase = self._phase
            if phase is None:
                return now
            frames = sys._current_frames()
            try:
                phase.samples += 1
                loop_frame = frames.get(self._loop_thread_ident)
                if loop_frame is None or _in_selector(loop_frame):
                    phase.idle_samples += 1
                    phase.current_busy = ("", 0.0)
                    loop_state = "loop idle"
                else:
                    loop_state = _loop_signature(loop_frame)
                    self._note_busy(phase, loop_state, now)
                if gap - self._interval >= self._freeze_gap:
                    phase.freezes.append((now - phase.started_at, gap, loop_state))
                self._sample_executor(phase, frames)
            finally:
                del frames
        return now

    @staticmethod
    def _note_busy(phase: PhaseRecord, signature: str, now: float) -> None:
        phase.busy[signature] += 1
        current, started_at = phase.current_busy
        if current != signature:
            phase.current_busy = (signature, now)
            return
        seconds = now - started_at
        if seconds > phase.longest_busy[1]:
            phase.longest_busy = (signature, seconds)

    @staticmethod
    def _sample_executor(phase: PhaseRecord, frames: dict[int, FrameType]) -> None:
        threads = [thread for thread in threading.enumerate() if thread.name.startswith(_EXECUTOR_THREAD_PREFIX)]
        if not threads:
            return
        phase.executor_threads_max = max(phase.executor_threads_max, len(threads))
        queues: dict[int, int] = {}
        for thread in threads:
            queue = futures_thread._threads_queues.get(thread)
            if queue is not None:
                queues[id(queue)] = queue.qsize()
            frame = frames.get(thread.ident) if thread.ident is not None else None
            if frame is None:
                continue
            signature = _executor_signature(frame)
            if signature is not None:
                phase.executor_busy[signature] += 1
        queued = max(queues.values(), default=0)
        phase.executor_queue_max = max(phase.executor_queue_max, queued)
        phase.executor_queue_total += queued
        if queued:
            phase.executor_queued_samples += 1


def format_report(phase: PhaseRecord, *, failed: bool) -> str:
    outcome = "failed" if failed else "passed"
    lines = [f"=== stall diagnostics: {phase.nodeid} ({outcome} after {phase.seconds:.1f}s in the call phase)"]
    if not phase.samples:
        lines.append("no samples: the phase ended before the sampler woke")
    else:
        idle = 100.0 * phase.idle_samples / phase.samples
        signature, seconds = phase.longest_busy
        stretch = f"{seconds:.1f}s in {signature}" if signature else "none"
        period = phase.seconds / phase.samples
        lines.append(
            f"loop thread: {phase.samples} samples, one every {period:.2f}s (asked for {phase.interval:g}s), "
            f"{idle:.0f}% in the selector; longest stretch out of it: {stretch}"
        )
        lines.extend(f"  {count:5d}x {signature}" for signature, count in phase.busy.most_common(4))
        if phase.executor_threads_max:
            queued = 100.0 * phase.executor_queued_samples / phase.samples
            mean = phase.executor_queue_total / phase.samples
            lines.append(
                f"default executor: up to {phase.executor_threads_max} threads; queue max {phase.executor_queue_max}, "
                f"non-empty in {queued:.0f}% of samples (mean {mean:.1f})"
            )
            lines.extend(
                f"  {count:5d} thread-samples {signature}" for signature, count in phase.executor_busy.most_common(6)
            )
        else:
            lines.append("default executor: no threads seen")
        if phase.freezes:
            lines.append(f"the sampler itself stalled {len(phase.freezes)}x (every thread stopped):")
            lines.extend(f"  {gap:.1f}s at +{at:.1f}s, then {state}" for at, gap, state in phase.freezes)
        else:
            lines.append("the sampler itself never stalled: the process kept running")
    # Collections are logged by the collecting thread, so they are known even when no sample was taken.
    if phase.gc_pauses:
        total = sum(seconds for _, _, seconds in phase.gc_pauses)
        pauses = ", ".join(
            f"gen{generation} {seconds:.1f}s at +{at:.1f}s" for at, generation, seconds in phase.gc_pauses[:8]
        )
        lines.append(f"gc pauses of {GC_PAUSE_SECONDS:g}s or more: {len(phase.gc_pauses)} ({total:.1f}s): {pauses}")
    else:
        lines.append(f"no gc pause of {GC_PAUSE_SECONDS:g}s or more")
    lines.append("=== end stall diagnostics")
    return "\n".join(lines) + "\n"


def _stderr_fileno() -> int | None:
    """The real stderr's descriptor, as pytest's faulthandler plugin finds it: xdist workers wrap ``sys.stderr``."""
    for stream in (sys.stderr, sys.__stderr__):
        if stream is None:
            continue
        try:
            fileno = stream.fileno()
        except AttributeError, ValueError, OSError:
            continue
        if fileno >= 0:
            return fileno
    return None


def start_stall_sampler_from_environment() -> StallSampler | None:
    """Start a sampler for this process when the flag is set; the report goes to the pre-capture stderr."""
    if not enabled():
        return None
    fileno = _stderr_fileno()
    if fileno is None:
        return None
    try:
        fd = os.dup(fileno)
    except OSError:
        return None

    def sink(text: str) -> None:
        os.write(fd, text.encode("utf-8", "replace"))

    sampler = _FdOwningSampler(fd=fd, loop_thread_ident=threading.main_thread().ident or 0, sink=sink)
    sampler.start()
    return sampler


class _FdOwningSampler(StallSampler):
    def __init__(self, *, fd: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fd = fd

    def close(self) -> None:
        super().close()
        os.close(self._fd)


# ------------------------------------------------------------------ Pilot waits


def awaited_chain(task: asyncio.Task[Any]) -> Iterator[Any]:
    """What a suspended task is nested in, outermost first: its coroutine, then what each one awaits."""
    awaited: Any = task.get_coro()
    while awaited is not None:
        yield awaited
        awaited = getattr(awaited, "cr_await", None) or getattr(awaited, "gi_yieldfrom", None)


def frame_of(awaited: Any) -> FrameType | None:
    """The frame a suspended coroutine or generator is stopped in; None for what ends a chain instead, such as
    a Future's iterator or an ``__await__`` wrapper."""
    return getattr(awaited, "cr_frame", None) or getattr(awaited, "gi_frame", None)


def _awaiting_chain(task: asyncio.Task[Any]) -> list[str]:
    """The coroutine frames a suspended task is nested in, outermost first, ending with what it awaits."""
    chain: list[str] = []
    for awaited in awaited_chain(task):
        frame = frame_of(awaited)
        if frame is None:
            chain.append(repr(awaited)[:120])
            break
        chain.append(_where(frame))
    return chain


def describe_pending_pumps(pumps: Iterable[MessagePump]) -> str:
    lines = []
    for pump in list(pumps)[:12]:
        task = pump._task
        state = (
            "no pump task"
            if task is None
            else ("pump task done" if task.done() else " -> ".join(_awaiting_chain(task)))
        )
        flags = "".join(flag for flag, on in (("closing ", pump._closing), ("pruning ", pump._pruning)) if on)
        aside = pump._pending_message
        held = f"+{type(aside).__name__} " if aside is not None else ""
        lines.append(f"  {pump!r}: {flags}queue={pump._message_queue.qsize()} {held}{state}")
    return "\n".join(lines)

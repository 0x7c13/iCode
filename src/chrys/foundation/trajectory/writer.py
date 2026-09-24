# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Single-writer, write-through trajectory event log with written-ack.

Contract:

* **Write-through** — ``emit`` returns only after the whole line reached the
  file (or after it is certain the event has no persistence guarantee). There
  is no "returned but still queued" state. Normal events are not ``fsync``ed;
  only checkpoints and the graceful-close tail are, and one of those answers
  DEGRADED when its ``fsync`` fails even though its line was written.
* **Ordered hand-off for synchronous producers** — a call site that cannot
  await (a ``finally`` in a sync function, a callback) uses ``submit`` +
  ``wait`` instead: ``submit`` takes the sequence and hands the encoded line
  to the worker before it returns, so ordering against every later ``emit``
  holds, and only the ack is awaited elsewhere. Such a producer learns
  nothing about persistence — but a later awaited ``WRITTEN`` still proves
  its line is on disk or explained by a gap.
* **Accounted prefix** — the worker writes strictly in ``sequence`` order. A
  ``WRITTEN`` ack for ``N`` proves every smaller sequence either has a line
  or is covered by a ``trajectory.gap`` persisted before ``N``.
* **Bounded wait, terminal degrade** — a producer waits at most
  ``write_ack_timeout``; a timeout or write failure flips the writer into
  ``TERMINAL_DEGRADED`` (one-way). Queued-not-started events become virtual
  slots with their payload released; the in-flight batch may finish and is
  confirmed line by line; later emits only register slots. Only gap/closure
  records may still be written, by the same worker, on the same descriptor.
* **Two-state result** — :class:`EmitResult.WRITTEN` carries the persistence
  guarantee; :class:`EmitResult.DEGRADED` carries none.

The worker is a dedicated daemon thread (never a ``ThreadPoolExecutor``): a
thread stuck in ``write()`` on a network disk must not keep the interpreter
from exiting, and the writer never opens a second descriptor or thread for
the same session.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import errno
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol

from chrys.foundation.trajectory.envelope import (
    INT64_MAX,
    LINE_BUDGET_BYTES,
    SYSTEM_ACTOR,
    EventDraft,
    TrajectoryEvent,
    build_event,
    check_int64_range,
    encode_event_line,
    malformed_id_pointers,
)
from chrys.foundation.trajectory.event_types import EventType, GapReason

logger = logging.getLogger(__name__)

DEFAULT_WRITE_ACK_TIMEOUT_SECONDS: Final = 3.0
DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS: Final = 2.0


class EmitResult(Enum):
    """Outcome of one ``emit``; only ``WRITTEN`` carries a persistence guarantee."""

    WRITTEN = "written"
    DEGRADED = "degraded"


class WriterState(Enum):
    """Writer lifecycle; ``TERMINAL_DEGRADED`` is one-way."""

    ACTIVE = "active"
    TERMINAL_DEGRADED = "terminal_degraded"
    CLOSED = "closed"


class WriteBackend(Protocol):
    """Minimal byte sink the worker thread drives; injectable for fault tests."""

    def write(self, data: memoryview) -> int:
        """Write some prefix of *data*; return the byte count (may be short)."""
        ...

    def fsync(self) -> None: ...

    def truncate(self, size: int) -> None: ...

    def close(self) -> None: ...


class FdWriteBackend:
    """Backend over a raw descriptor from ``secure_open_owner_only_append``."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    @property
    def fd(self) -> int:
        return self._fd

    def write(self, data: memoryview) -> int:
        return os.write(self._fd, data)

    def fsync(self) -> None:
        os.fsync(self._fd)

    def truncate(self, size: int) -> None:
        os.ftruncate(self._fd, size)

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class DroppedRange:
    """One immutable range of confirmed-unwritten sequences."""

    first_sequence: int
    last_sequence: int
    dropped_count: int


@dataclass(slots=True)
class _Pending:
    sequence: int
    data: bytes | None
    future: concurrent.futures.Future[EmitResult]
    fsync_after: bool = False
    system: bool = False
    covers: tuple[DroppedRange, ...] = ()
    """For gap records: compact ranges this gap accounts for."""


@dataclass(slots=True)
class WriterSnapshot:
    """Point-in-time view for diagnostics and tests."""

    state: WriterState
    next_sequence: int
    last_written_sequence: int
    last_durable_sequence: int
    uncovered_slots: tuple[int, ...]
    uncovered_ranges: tuple[DroppedRange, ...]
    inflight_sequences: tuple[int, ...]
    queued_sequences: tuple[int, ...]
    degraded_reason: str | None
    tail_torn: bool
    runtime_closed: bool
    written_count: int = 0
    written_bytes: int = 0


@dataclass(slots=True)
class _GapBookkeeping:
    pending_by_gap_sequence: dict[int, tuple[DroppedRange, ...]] = field(default_factory=dict)
    attempt_failed: bool = False


@dataclass(frozen=True, slots=True)
class SubmittedEvent:
    """A queued line or the immediate result when nothing was queued (degraded/gap)."""

    result: _Pending | EmitResult


class TrajectoryWriter:
    """Per-session writer: one thread, one descriptor, one sequence counter."""

    def __init__(
        self,
        *,
        backend: WriteBackend,
        session_id: str,
        runtime_id: str,
        coverage_id: str,
        branch_id: str,
        initial_sequence: int = 0,
        initial_offset: int = 0,
        recovered_gap: tuple[int, int] | None = None,
        write_ack_timeout: float = DEFAULT_WRITE_ACK_TIMEOUT_SECONDS,
        on_degraded: Callable[[str], None] | None = None,
        on_worker_exit: Callable[[], None] | None = None,
        thread_name: str | None = None,
    ) -> None:
        if write_ack_timeout <= 0:
            raise ValueError("write_ack_timeout must be positive.")
        self._backend = backend
        self._session_id = session_id
        self._runtime_id = runtime_id
        self._coverage_id = coverage_id
        self._branch_id = branch_id
        self._write_ack_timeout = write_ack_timeout
        self._on_degraded = on_degraded
        self._on_worker_exit = on_worker_exit

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: deque[_Pending] = deque()
        self._inflight: list[_Pending] = []
        self._state = WriterState.ACTIVE
        self._frozen = False
        self._stop_requested = False
        self._next_sequence = initial_sequence + 1
        self._last_written = initial_sequence
        self._last_durable = 0
        self._confirmed_offset = initial_offset
        self._uncovered_slots: list[int] = []
        self._uncovered_ranges: list[DroppedRange] = []
        self._gap = _GapBookkeeping()
        self._degraded_reason: str | None = None
        self._tail_torn = False
        self._runtime_closed = False
        self._written_count = 0
        self._written_bytes = 0
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name or f"chrys-trajectory-writer-{session_id[:8]}",
            daemon=True,
        )
        self._started = False
        self._resources_released = False
        self._abort_truncate_to: int | None = None
        self._abort_rewound = False
        if recovered_gap is not None:
            # Queued here rather than on ``start``: the first emit starts the
            # worker too, so the only place this gap is guaranteed to precede
            # every line of the resumed runtime is before anyone holds the
            # writer at all.
            first, last = recovered_gap
            if 1 <= first <= last:
                with self._lock:
                    recovered = DroppedRange(
                        first_sequence=first,
                        last_sequence=last,
                        dropped_count=last - first + 1,
                    )
                    self._queue_gap_locked(
                        first=first,
                        last=last,
                        reason=GapReason.RECOVERED_UNREADABLE,
                        covers=(recovered,),
                    )

    # ------------------------------------------------------------------ props

    @property
    def state(self) -> WriterState:
        return self._state

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def coverage_id(self) -> str:
        return self._coverage_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def write_ack_timeout(self) -> float:
        return self._write_ack_timeout

    @property
    def is_degraded(self) -> bool:
        return self._state is WriterState.TERMINAL_DEGRADED

    @property
    def runtime_closed(self) -> bool:
        return self._runtime_closed

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def last_assigned_sequence(self) -> int:
        """The highest sequence handed out so far (0 before the first event)."""
        with self._lock:
            return self._next_sequence - 1

    def snapshot(self) -> WriterSnapshot:
        with self._lock:
            return WriterSnapshot(
                state=self._state,
                next_sequence=self._next_sequence,
                last_written_sequence=self._last_written,
                last_durable_sequence=self._last_durable,
                uncovered_slots=tuple(self._uncovered_slots),
                uncovered_ranges=tuple(self._uncovered_ranges),
                inflight_sequences=tuple(p.sequence for p in self._inflight),
                queued_sequences=tuple(p.sequence for p in self._queue),
                degraded_reason=self._degraded_reason,
                tail_torn=self._tail_torn,
                runtime_closed=self._runtime_closed,
                written_count=self._written_count,
                written_bytes=self._written_bytes,
            )

    def set_branch_id(self, branch_id: str) -> None:
        """Switch the branch stamped on subsequent events (rollback)."""
        with self._lock:
            self._branch_id = branch_id

    def start(self) -> None:
        """Start the worker thread (idempotent; the first emit also starts it)."""
        with self._lock:
            if self._started:
                return
            # Only a thread that actually started may be joined later: claiming
            # one that the system refused would make ``_stop_worker`` raise on
            # the join and strand the descriptor and the session's lease.
            self._thread.start()
            self._started = True

    def measure_line(self, draft: EventDraft) -> int:
        """Return the encoded line length *draft* would have at the next sequence."""
        event = build_event(
            draft,
            sequence=10**9,
            runtime_id=self._runtime_id,
            coverage_id=self._coverage_id,
            session_id=self._session_id,
            branch_id=self._branch_id,
        )
        return len(encode_event_line(event))

    # ------------------------------------------------------------------ emit

    def submit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> SubmittedEvent:
        """Assign the sequence and queue the line now; the ack is awaited separately.

        For producers that cannot await at the point the event happens (a
        synchronous stream hook) but must keep write order: the sequence is
        taken synchronously here, so a later ``emit`` from the same task
        always lands after it. Pass the result to :meth:`wait` to observe
        the ack and keep the bounded-wait contract.
        """
        return SubmittedEvent(self._prepare(draft, payload_factory=payload_factory))

    async def wait(self, submitted: SubmittedEvent) -> EmitResult:
        """Await the ack of a :meth:`submit` (timeout flips the writer degraded)."""
        result = submitted.result
        if isinstance(result, EmitResult):
            return result
        return await self._await_ack(result)

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        """Write one event through; returns ``WRITTEN`` only once the line is in the file.

        The wait is bounded by ``write_ack_timeout``; the write request is
        shielded from the caller's cancellation (a cancelled producer's line is
        still written). Timeout flips the writer into ``TERMINAL_DEGRADED``.

        ``payload_factory`` builds the payload from the assigned sequence for
        events whose payload refers to their own position (``session.rollback``
        closes its superseded range at ``sequence - 1``).
        """
        prepared = self._prepare(draft, payload_factory=payload_factory)
        if isinstance(prepared, EmitResult):
            return prepared
        return await self._await_ack(prepared)

    def emit_blocking(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        """Synchronous ``emit`` for threads that do not run an event loop."""
        prepared = self._prepare(draft, payload_factory=payload_factory)
        if isinstance(prepared, EmitResult):
            return prepared
        return self._wait_ack_blocking(prepared)

    async def checkpoint(self) -> EmitResult:
        """Write ``trajectory.checkpoint`` and ``fsync``; ``last_durable`` advances here only."""

        def payload(sequence: int) -> dict[str, Any]:
            return {
                "last_assigned": sequence,
                "last_written": self._last_written,
                "last_durable": self._last_durable,
            }

        prepared = self._prepare(
            EventDraft(event_type=EventType.CHECKPOINT, actor=SYSTEM_ACTOR),
            payload_factory=payload,
            fsync_after=True,
        )
        if isinstance(prepared, EmitResult):
            return prepared
        return await self._await_ack(prepared)

    async def close(self, *, reason: str, join_timeout: float = DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS) -> bool:
        """Graceful close: freeze → (final gap) → coverage.ended → runtime.finished → fsync → close.

        Returns ``True`` when the runtime was closed cleanly (both closure
        markers ``WRITTEN`` and durable). Any failure leaves the runtime
        unclosed on disk, which is exactly the evidence a reader needs.
        """
        with self._lock:
            self._frozen = True
        closed = False
        try:
            if self._state is WriterState.ACTIVE:
                closed = await self._close_active(reason)
            elif self._state is WriterState.TERMINAL_DEGRADED:
                closed = await self._close_degraded(reason)
        except Exception:
            # Closing is the last thing this writer does, so a failure here has
            # nowhere left to be handled — and an unclosed runtime on disk is
            # already how that reads. What must not be lost is the handback
            # below: the descriptor and the session's lease come back either
            # way, or the session stays leased for the rest of the process.
            logger.warning("Trajectory writer close failed for session %s", self._session_id, exc_info=True)
        finally:
            self._stop_worker(join_timeout)
        return closed

    def abort(self, *, join_timeout: float = DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS) -> bool:
        """Erase and stop an unannounced runtime without writing closure markers.

        Activation owns this path: if its prelude could not complete, emitting
        ``coverage.ended`` or ``runtime.finished`` would create terminals for a
        runtime readers never saw start. The worker rewinds every prelude byte
        it may have written before releasing its descriptor and lease, allowing
        activation to retry from the exact prefix it originally recovered.
        """
        with self._lock:
            self._frozen = True
            self._abort_truncate_to = self._confirmed_offset - self._written_bytes
        self._stop_worker(join_timeout)
        return not self.thread_alive and self._abort_rewound

    # --------------------------------------------------------------- internals

    def _prepare(
        self,
        draft: EventDraft,
        *,
        payload_factory: Callable[[int], Mapping[str, Any]] | None = None,
        fsync_after: bool = False,
        system: bool = False,
        covers: tuple[DroppedRange, ...] = (),
    ) -> _Pending | EmitResult:
        with self._lock:
            if (self._frozen and not system) or self._stop_requested:
                return EmitResult.DEGRADED
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._state is not WriterState.ACTIVE and not system:
                # Virtual slot only: no payload is retained, no wait is taken.
                self._uncovered_slots.append(sequence)
                return EmitResult.DEGRADED
            try:
                payload = payload_factory(sequence) if payload_factory is not None else None
                event = build_event(
                    draft if payload is None else _with_payload(draft, payload),
                    sequence=sequence,
                    runtime_id=self._runtime_id,
                    coverage_id=self._coverage_id,
                    session_id=self._session_id,
                    branch_id=self._branch_id,
                )
                encoded = _encode_checked(event)
            except Exception:
                # The slot is spent the moment it is handed out. A factory that
                # raises — or a draft that cannot be built into an event —
                # leaves nothing to write, which is the encode failure below by
                # another route: letting it out of here instead would drop the
                # sequence from the prefix with no gap to explain it.
                logger.warning(
                    "Trajectory event for sequence %d could not be built; recording a gap", sequence, exc_info=True
                )
                encoded = GapReason.ENCODE_FAILURE
            if isinstance(encoded, str):
                # The event itself cannot be written (over budget, out of range,
                # unencodable). Its slot is accounted for by a gap that takes the
                # next sequence, so the prefix stays explained.
                self._uncovered_slots.append(sequence)
                self._enqueue_gap_locked(reason=encoded)
                return EmitResult.DEGRADED
            pending = _Pending(
                sequence=sequence,
                data=encoded,
                future=concurrent.futures.Future(),
                fsync_after=fsync_after,
                system=system,
                covers=covers,
            )
            self._queue.append(pending)
            self._cond.notify()
            if not self._started:
                self._thread.start()
                self._started = True
        return pending

    async def _await_ack(self, pending: _Pending) -> EmitResult:
        loop = asyncio.get_running_loop()
        wrapped = asyncio.wrap_future(pending.future, loop=loop)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self._write_ack_timeout)
        except TimeoutError:
            # The worker resolves the concurrent future from its own thread;
            # ``wrap_future`` then schedules delivery onto this event loop.
            # Under scheduler pressure the timeout callback can run before
            # that delivery callback even though the write already finished.
            # Read the worker-owned result directly before declaring the
            # writer degraded so an event-loop delay cannot turn a persisted
            # lifecycle terminal into a virtual slot.
            if pending.future.done():
                return pending.future.result()
            self._enter_terminal_degraded(GapReason.WRITE_TIMEOUT)
            return EmitResult.DEGRADED

    def _wait_ack_blocking(self, pending: _Pending) -> EmitResult:
        try:
            return pending.future.result(timeout=self._write_ack_timeout)
        except concurrent.futures.TimeoutError:
            self._enter_terminal_degraded(GapReason.WRITE_TIMEOUT)
            return EmitResult.DEGRADED

    def _enter_terminal_degraded(self, reason: str) -> None:
        dropped: list[_Pending] = []
        with self._lock:
            if self._state is not WriterState.ACTIVE:
                return
            self._state = WriterState.TERMINAL_DEGRADED
            self._degraded_reason = reason
            # Queue classification at the transition: everything still in the
            # queue is "queued, not started" — the worker moves a batch out of
            # the queue under this lock before touching the descriptor, so the
            # in-flight batch is never here. Those slots are confirmed
            # unwritten; release their payload and answer their waiters now.
            while self._queue:
                item = self._queue.popleft()
                item.data = None
                if item.covers:
                    self._uncovered_ranges.extend(self._gap.pending_by_gap_sequence.pop(item.sequence, item.covers))
                self._uncovered_slots.append(item.sequence)
                dropped.append(item)
            self._uncovered_slots.sort()
            self._cond.notify_all()
        for item in dropped:
            _resolve(item.future, EmitResult.DEGRADED)
        logger.warning(
            "Trajectory writer for session %s entered TERMINAL_DEGRADED (%s); "
            "later events carry no persistence guarantee.",
            self._session_id,
            reason,
        )
        if self._on_degraded is not None:
            with contextlib.suppress(Exception):
                self._on_degraded(reason)

    def _enqueue_gap_locked(self, *, reason: str) -> list[_Pending]:
        """Queue one gap per contiguous run of uncovered slots (lock held)."""
        if self._tail_torn or not (self._uncovered_slots or self._uncovered_ranges):
            return []
        if self._next_sequence > INT64_MAX:
            # The gap would have to carry a sequence no line may carry, and it
            # is built past the check that refuses an ordinary event for that
            # — it would land as a line no reader can decode. Nothing after
            # this slot is writable either, so the file ends here: an unwritten
            # slot at the end of a file reads as a runtime that stopped, which
            # is a shape readers already handle.
            return []
        ranges = _coalesced_dropped_ranges(
            [
                *self._uncovered_ranges,
                *(
                    DroppedRange(first_sequence=run[0], last_sequence=run[-1], dropped_count=len(run))
                    for run in _contiguous_runs(self._uncovered_slots)
                ),
            ]
        )
        self._uncovered_slots = []
        self._uncovered_ranges = []
        return [
            self._queue_gap_locked(
                first=dropped.first_sequence,
                last=dropped.last_sequence,
                reason=reason,
                covers=(dropped,),
            )
            for dropped in ranges
        ]

    def _queue_gap_locked(
        self,
        *,
        first: int,
        last: int,
        reason: str,
        covers: tuple[DroppedRange, ...],
    ) -> _Pending:
        """Queue one gap over ``[first, last]`` (lock held).

        ``covers`` keeps compact intervals to hand back if this gap is never
        written, so the next one explains them again. Recovered corruption can
        claim an enormous sequence range; retaining intervals here avoids the
        unbounded ``range`` expansion that per-slot bookkeeping would require.
        """
        sequence = self._next_sequence
        self._next_sequence += 1
        draft = EventDraft(
            event_type=EventType.GAP,
            actor=SYSTEM_ACTOR,
            payload={
                "first_sequence": first,
                "last_sequence": last,
                "dropped_count": last - first + 1,
                "reason": reason,
            },
        )
        event = build_event(
            draft,
            sequence=sequence,
            runtime_id=self._runtime_id,
            coverage_id=self._coverage_id,
            session_id=self._session_id,
            branch_id=self._branch_id,
        )
        pending = _Pending(
            sequence=sequence,
            data=encode_event_line(event),
            future=concurrent.futures.Future(),
            system=True,
            covers=covers,
        )
        if covers:
            self._gap.pending_by_gap_sequence[sequence] = covers
        self._queue.append(pending)
        self._cond.notify()
        return pending

    # --------------------------------------------------------------- worker

    def _run(self) -> None:
        try:
            self._drain()
        finally:
            # The descriptor and the session's writer lease belong to this
            # thread for as long as it may still write, so this thread is what
            # hands them back — on request, or whenever a blocked ``write()``
            # finally lets go. Nothing outside can observe that moment without
            # a thread of its own, and a thread is exactly what a system out
            # of them cannot give.
            self._release_worker_resources()

    def _drain(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop_requested:
                    self._cond.wait()
                if not self._queue and self._stop_requested:
                    return
                batch = list(self._queue)
                self._queue.clear()
                self._inflight = batch
            self._write_batch(batch)
            with self._lock:
                self._inflight = []
                # ``CLOSED`` counts too: a close that timed out on this thread
                # moved the state on and left, and the slots it could not cover
                # are no less lost for that. This thread still holds the file,
                # so it is the only place left that can say so — without it the
                # next runtime resumes on a slot that was handed out already
                # and nothing on disk records that an event went missing.
                settling = self._state in (WriterState.TERMINAL_DEGRADED, WriterState.CLOSED)
                if settling and not self._gap.attempt_failed and (self._uncovered_slots or self._uncovered_ranges):
                    self._enqueue_gap_locked(reason=self._degraded_reason or GapReason.WRITE_FAILURE)

    def _write_batch(self, batch: list[_Pending]) -> None:
        payload = b"".join(item.data or b"" for item in batch)
        view = memoryview(payload)
        written = 0
        failure: BaseException | None = None
        try:
            while written < len(payload):
                count = self._backend.write(view[written:])
                if not isinstance(count, int) or count <= 0:
                    raise OSError(errno.EIO, "Trajectory backend wrote nothing.")
                written += count
        except BaseException as exc:  # a stuck/failing backend must never kill the thread
            failure = exc
        if failure is None:
            self._complete_batch(batch, written=written)
            return
        self._fail_batch(batch, written=written, failure=failure)

    def _complete_batch(self, batch: list[_Pending], *, written: int) -> None:
        fsync_failure: BaseException | None = None
        needs_fsync = any(item.fsync_after for item in batch)
        if needs_fsync:
            try:
                self._backend.fsync()
            except BaseException as exc:
                fsync_failure = exc
        to_resolve: list[tuple[_Pending, EmitResult]] = []
        with self._lock:
            self._confirmed_offset += written
            for item in batch:
                self._last_written = item.sequence
                self._written_count += 1
                self._written_bytes += len(item.data or b"")
                if item.covers:
                    self._gap.pending_by_gap_sequence.pop(item.sequence, None)
                item.data = None
                # Every line reached the file. An event that asked to be made
                # durable asked for more than that, so a failed ``fsync`` is
                # answered with DEGRADED: the only callers that set the flag
                # are the checkpoint and the closure tail, and both must be
                # able to tell "written" from "written and durable".
                durable_denied = fsync_failure is not None and item.fsync_after
                to_resolve.append((item, EmitResult.DEGRADED if durable_denied else EmitResult.WRITTEN))
            if needs_fsync and fsync_failure is None:
                self._last_durable = batch[-1].sequence
        for item, result in to_resolve:
            _resolve(item.future, result)
        if fsync_failure is not None:
            logger.warning("Trajectory checkpoint fsync failed for session %s: %s", self._session_id, fsync_failure)

    def _fail_batch(self, batch: list[_Pending], *, written: int, failure: BaseException) -> None:
        to_resolve: list[tuple[_Pending, EmitResult]] = []
        truncate_to: int | None = None
        with self._lock:
            offset = 0
            confirmed = 0
            torn = False
            for item in batch:
                size = len(item.data or b"")
                end = offset + size
                if end <= written:
                    confirmed = end
                    self._last_written = item.sequence
                    self._written_count += 1
                    self._written_bytes += size
                    if item.covers:
                        self._gap.pending_by_gap_sequence.pop(item.sequence, None)
                    to_resolve.append((item, EmitResult.WRITTEN))
                else:
                    if offset < written:
                        torn = True
                    if item.covers:
                        self._uncovered_ranges.extend(self._gap.pending_by_gap_sequence.pop(item.sequence, ()))
                        self._gap.attempt_failed = True
                    self._uncovered_slots.append(item.sequence)
                    to_resolve.append((item, EmitResult.DEGRADED))
                item.data = None
                offset = end
            self._uncovered_slots.sort()
            self._confirmed_offset += confirmed
            if torn:
                truncate_to = self._confirmed_offset
        # Transition first so a producer that observes DEGRADED already sees the
        # terminal state (and the on_degraded callback has fired).
        self._enter_terminal_degraded(GapReason.WRITE_FAILURE)
        logger.warning("Trajectory write failed for session %s: %s", self._session_id, failure)
        for item, result in to_resolve:
            _resolve(item.future, result)
        if truncate_to is not None:
            # Repair our own torn tail so a later gap/closure line does not land
            # after a half-written line (which would be mid-file corruption).
            try:
                self._backend.truncate(truncate_to)
            except BaseException as exc:
                with self._lock:
                    self._tail_torn = True
                logger.warning("Trajectory torn-tail repair failed for session %s: %s", self._session_id, exc)

    # --------------------------------------------------------------- closing

    async def _close_active(self, reason: str) -> bool:
        ended = await self._emit_system(
            EventType.COVERAGE_ENDED,
            payload_factory=lambda seq: {"coverage_id": self._coverage_id, "last_sequence": seq - 1},
        )
        if ended is not EmitResult.WRITTEN:
            return False
        finished = await self._emit_system(
            EventType.RUNTIME_FINISHED,
            payload_factory=lambda _seq: {"reason": reason},
            fsync_after=True,
        )
        if finished is not EmitResult.WRITTEN:
            return False
        with self._lock:
            self._runtime_closed = True
        return True

    async def _close_degraded(self, reason: str) -> bool:
        # ① emit is frozen (done by close()). ② wait for the in-flight batch to
        # settle — a batch that never returns leaves slots neither written nor
        # dropped, so no final gap can be constructed and the runtime stays open.
        deadline = time.monotonic() + self._write_ack_timeout
        while True:
            with self._lock:
                inflight = bool(self._inflight)
                torn = self._tail_torn
            if not inflight:
                break
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)
        if torn:
            return False
        with self._lock:
            self._gap.attempt_failed = False
            gaps = self._enqueue_gap_locked(reason=GapReason.DEGRADED_CLOSE)
        for pending in gaps:
            if await self._await_ack(pending) is not EmitResult.WRITTEN:
                return False
        with self._lock:
            if self._uncovered_slots or self._uncovered_ranges:
                return False
        return await self._close_active(reason)

    async def _emit_system(
        self,
        event_type: str,
        *,
        payload_factory: Callable[[int], dict[str, Any]],
        fsync_after: bool = False,
    ) -> EmitResult:
        prepared = self._prepare(
            EventDraft(event_type=event_type, actor=SYSTEM_ACTOR),
            payload_factory=payload_factory,
            fsync_after=fsync_after,
            system=True,
        )
        if isinstance(prepared, EmitResult):
            return prepared
        return await self._await_ack(prepared)

    def _stop_worker(self, join_timeout: float) -> None:
        with self._lock:
            self._stop_requested = True
            self._cond.notify_all()
        if self._started:
            self._thread.join(timeout=max(0.0, join_timeout))
        if self._thread.is_alive():
            # A stuck worker keeps its descriptor: closing it underneath a
            # blocked write() could let the number be reused by another file.
            logger.warning(
                "Trajectory writer thread for session %s did not exit within %.1fs; abandoning it.",
                self._session_id,
                join_timeout,
            )
            with self._lock:
                self._state = WriterState.CLOSED
            # The worker hands its own resources back when it returns.
            return
        with self._lock:
            self._state = WriterState.CLOSED
        self._release_worker_resources()

    def _release_worker_resources(self) -> None:
        """Close the descriptor and hand the lease back; runs at most once.

        Reached from the worker's own exit and from ``_stop_worker`` (which
        covers a worker that never started), so it has to settle who does it.
        """
        with self._lock:
            if self._resources_released:
                return
            self._resources_released = True
            callback = self._on_worker_exit
            self._on_worker_exit = None
            truncate_to = self._abort_truncate_to
        if truncate_to is not None:
            try:
                self._backend.truncate(truncate_to)
                self._backend.fsync()
                self._abort_rewound = True
            except Exception:
                logger.warning("Trajectory activation rollback failed for session %s", self._session_id, exc_info=True)
        with contextlib.suppress(Exception):
            self._backend.close()
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.debug("Trajectory writer exit callback failed", exc_info=True)


def _with_payload(draft: EventDraft, payload: Mapping[str, Any]) -> EventDraft:
    return EventDraft(
        event_type=draft.event_type,
        event_id=draft.event_id,
        actor=draft.actor,
        payload=payload,
        measurements=draft.measurements,
        turn_id=draft.turn_id,
        operation_id=draft.operation_id,
        parent_operation_id=draft.parent_operation_id,
        links=draft.links,
        segmented_fields=draft.segmented_fields,
        occurred_at=draft.occurred_at,
        monotonic_ns=draft.monotonic_ns,
    )


def _encode_checked(event: TrajectoryEvent) -> bytes | str:
    """Encode *event*; return the line bytes or a gap reason code on failure."""
    data = event.to_dict()
    if check_int64_range(data):
        return GapReason.VALUE_OUT_OF_RANGE
    if malformed_id_pointers(data):
        # A line no reader can decode would read as corruption; a gap says
        # exactly what happened and keeps the accounted prefix honest.
        return GapReason.ENCODE_FAILURE
    try:
        line = encode_event_line(event)
    except TypeError, ValueError:
        return GapReason.ENCODE_FAILURE
    if len(line) > LINE_BUDGET_BYTES:
        return GapReason.LINE_BUDGET_EXCEEDED
    return line


def _contiguous_runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in sorted(set(values)):
        if runs and runs[-1][-1] + 1 == value:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _coalesced_dropped_ranges(ranges: list[DroppedRange]) -> list[DroppedRange]:
    """Merge overlapping/adjacent dropped intervals without expanding them."""
    merged: list[DroppedRange] = []
    for dropped in sorted(ranges, key=lambda item: (item.first_sequence, item.last_sequence)):
        if merged and dropped.first_sequence <= merged[-1].last_sequence + 1:
            first = merged[-1].first_sequence
            last = max(merged[-1].last_sequence, dropped.last_sequence)
            merged[-1] = DroppedRange(
                first_sequence=first,
                last_sequence=last,
                dropped_count=last - first + 1,
            )
        else:
            merged.append(dropped)
    return merged


def _resolve(future: concurrent.futures.Future[EmitResult], result: EmitResult) -> None:
    with contextlib.suppress(Exception):
        if not future.done():
            future.set_result(result)

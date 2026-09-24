# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Restore-only repair of interrupted runs; the log alone owns terminal facts.

A read-only tail check skips finished runs. Unfinished logs are rechecked under
one reconciliation lock through the descriptor that receives the orphan terminal.
Torn tails are separated before append. No recovery events are published live.
"""

from __future__ import annotations

import errno
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from chrys.foundation.platform.files import secure_open_owner_only_append, secure_open_owner_verified_binary
from chrys.foundation.trajectory.envelope import (
    EnvelopeError,
    EventDraft,
    TrajectoryEvent,
    build_event,
    decode_event_line,
    encode_event_line,
    peek_envelope_header,
)
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.util.lock import FileLock
from chrys.service.workflows.layout import EVENTS_FILE, iter_run_dirs
from chrys.service.workflows.outcomes import ORPHAN_REASON_PROCESS_TERMINATED, RunOutcome
from chrys.service.workflows.store import WORKFLOW_EVENT_TYPES, RunRecord, RunTerminal, read_run_header

logger = logging.getLogger(__name__)

MAX_EVENTS_BYTES: Final = 256 * 1024 * 1024
"""Bounds reverse scanning, including malformed or sparse files, without allocating the whole log."""
_TAIL_BLOCK_BYTES: Final = 64 * 1024
RECONCILE_LOCK_FILE: Final = ".reconcile.lock"
RECONCILE_LOCK_TIMEOUT: Final = 30.0  # seconds


@dataclass(frozen=True, slots=True)
class OrphanReconciliation:
    """What one pass did: runs given a terminal, runs already terminal, runs it could not judge."""

    reconciled: tuple[str, ...] = ()
    already_finished: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    truncated: bool = False


def reconcile_orphaned_runs(session_dir: Path) -> OrphanReconciliation:
    """Give every unfinished run under *session_dir* its ``orphaned`` terminal; idempotent."""
    run_dirs, truncated = iter_run_dirs(session_dir)
    reconciled: list[str] = []
    finished: list[str] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        try:
            outcome = _reconcile_run(run_dir)
        except OSError, ValueError:  # EnvelopeError is a ValueError
            if not run_dir.exists():  # the session was deleted under a reconciliation a cancelled restore left running
                logger.debug("workflow run %s was deleted during reconciliation", run_dir.name)
            else:
                logger.warning("workflow run %s could not be reconciled", run_dir.name, exc_info=True)
            skipped.append(run_dir.name)
            continue
        (reconciled if outcome else finished).append(run_dir.name)
    return OrphanReconciliation(tuple(reconciled), tuple(finished), tuple(skipped), truncated)


def read_run_terminal(run_dir: Path) -> RunTerminal | None:
    """Read a durable terminal without repairing files or acquiring the session's execution lock."""
    try:
        with secure_open_owner_verified_binary(run_dir / EVENTS_FILE) as handle:
            _last, record = _last_complete(_reverse_complete_lines(handle))
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise
    return _terminal(record) if record is not None and record.event_type == RunRecord.RUN_FINISHED else None


def _reverse_complete_lines(handle: BinaryIO) -> Iterator[bytes]:
    """Scan backward from one EOF snapshot; ignore its torn tail, and stop on demand.

    Only a line crossing block boundaries is retained. The existing ceiling
    bounds pathological scans, not file size: a large live log needs one tail
    block in the ordinary case. Joining a split line once avoids quadratic
    copying for large records.
    """
    position = handle.seek(0, os.SEEK_END)
    scanned = 0
    fragments: list[bytes] = []
    torn_tail = True
    while position:
        if scanned == MAX_EVENTS_BYTES:
            raise ValueError(f"{EVENTS_FILE} tail exceeds the {MAX_EVENTS_BYTES}-byte scan ceiling")
        length = min(position, _TAIL_BLOCK_BYTES, MAX_EVENTS_BYTES - scanned)
        position -= length
        handle.seek(position)
        pieces = handle.read(length).split(b"\n")
        scanned += length
        for piece in reversed(pieces[1:]):
            if torn_tail:
                torn_tail = False
            else:
                yield piece + b"".join(reversed(fragments))
            fragments.clear()
        if not torn_tail:
            fragments.append(pieces[0])
    if not torn_tail:
        yield b"".join(reversed(fragments))


def _terminal(record: TrajectoryEvent) -> RunTerminal:
    reason = record.payload.get("reason")
    return RunTerminal(
        outcome=str(record.payload.get("outcome", "")),
        last_seq=record.sequence,
        reason=reason if isinstance(reason, str) else "",
        finished_at=record.occurred_at,
    )


def _reconcile_run(run_dir: Path) -> bool:
    """Append once, without creating any directories if deletion races restoration."""
    if read_run_terminal(run_dir) is not None:
        return False
    with FileLock(run_dir / RECONCILE_LOCK_FILE, timeout=RECONCILE_LOCK_TIMEOUT):
        return _reconcile_run_locked(run_dir)


def _reconcile_run_locked(run_dir: Path) -> bool:
    header = read_run_header(run_dir)
    run_id = header.get("run_id")
    session_id = header.get("session_id")
    if not isinstance(run_id, str) or not run_id or not isinstance(session_id, str):
        raise ValueError("run header lacks its identity")
    events_path = run_dir / EVENTS_FILE
    handle = secure_open_owner_only_append(events_path)
    try:
        # Read through the descriptor being appended to: the file examined is the file written.
        with os.fdopen(os.dup(handle.fd), "rb") as reader:
            last, last_record = _last_complete(_reverse_complete_lines(reader))
            if last_record is not None and last_record.event_type == RunRecord.RUN_FINISHED:
                return False
            sequence = (last.sequence if last is not None else _highest_sequence(_reverse_complete_lines(reader))) + 1
            size = reader.seek(0, os.SEEK_END)
            if size:
                reader.seek(-1, os.SEEK_END)
            tail = bool(size and reader.read(1) != b"\n")
        draft = EventDraft(
            event_type=RunRecord.RUN_FINISHED,
            payload={"outcome": RunOutcome.ORPHANED.value, "reason": ORPHAN_REASON_PROCESS_TERMINATED},
        )
        event = build_event(
            draft,
            sequence=sequence,
            runtime_id=run_id,
            coverage_id=last.coverage_id if last is not None else new_analytics_id(),
            session_id=session_id,
            branch_id=last.branch_id if last is not None else new_analytics_id(),
        )
        line = (b"\n" if tail else b"") + encode_event_line(event)
        _write_all(handle.fd, line)
        os.fsync(handle.fd)
    finally:
        os.close(handle.fd)
    return True


def _last_complete(lines: Iterable[bytes]) -> tuple[TrajectoryEvent | None, TrajectoryEvent | None]:
    """Read newest-first lines for the last envelope and last workflow record.

    The two differ when the store closed its writer after the terminal: the
    runtime markers that follow ``workflow.log.run.finished`` are the writer's,
    not the run's.
    """
    last: TrajectoryEvent | None = None
    for raw in lines:
        try:
            event = decode_event_line(raw)
        except EnvelopeError:
            continue
        if last is None:
            last = event
        if event.event_type in WORKFLOW_EVENT_TYPES:
            return last, event
    return last, None


def _highest_sequence(lines: Iterable[bytes]) -> int:
    highest = 0
    for raw in lines:
        header = peek_envelope_header(raw)
        if header is not None:
            highest = max(highest, header.sequence)
    return highest


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]

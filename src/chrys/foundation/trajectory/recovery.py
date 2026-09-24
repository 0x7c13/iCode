# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Activation-time recovery for an existing events file.

A crash can leave a torn final line. Appending after it would turn the torn
bytes into a bad line in the *middle* of the file, where "ignore the last
line" no longer applies. So before a writer resumes a file it must, under the
session's exclusive lock: locate the last complete newline, truncate the
torn tail, and recover ``sequence`` from the last valid event.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from chrys.foundation.trajectory.envelope import (
    SCHEMA_VERSION,
    EnvelopeError,
    TrajectoryEvent,
    decode_event_line,
    peek_envelope_header,
)

_INITIAL_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class RecoveryScan:
    """What the tail of an existing events file looks like."""

    file_size: int
    complete_offset: int
    """Byte offset just past the last complete newline (0 for an empty file)."""
    last_sequence: int
    """Highest slot the file has spoken for (0 when it has spoken for none).

    Usually the last valid event's, but a corrupt line whose header still
    names its slot counts too: those bytes stay on disk, so handing the same
    sequence out again would put two lines on one slot.
    """
    last_event: TrajectoryEvent | None
    had_valid_events: bool
    newer_schema_version: int | None = None
    """Set when the tail was written under a schema this build must not append after."""
    unreadable_slots: tuple[int, int] | None = None
    """Inclusive slot range the file spent on lines no reader can show, if any.

    A damaged line keeps its bytes and its slot but never reaches a reader,
    so resuming past it leaves a hole in the sequence that nothing explains.
    A writer that resumes here owes the file a gap over this range first.
    """
    unreadable_tail: bool = False
    """Set when a complete line past the last valid event does not even name its slot.

    A damaged line whose header still parses spends a slot this build can
    name, and ``unreadable_slots`` accounts for it. One whose header does not
    parse spends a slot nobody can name — and a slot that cannot be named
    cannot be skipped over, because the next sequence to hand out is exactly
    the thing in question. There is nothing to resume from here.
    """

    @property
    def truncated_bytes(self) -> int:
        return self.file_size - self.complete_offset


def scan_for_recovery(path: Path) -> RecoveryScan:
    """Inspect the tail of *path* without modifying it."""
    with path.open("rb") as handle:
        return scan_open_file(handle.fileno())


def scan_open_file(fd: int) -> RecoveryScan:
    """Inspect the tail of the file *fd* is open on, without modifying it.

    This is how a writer scans: the descriptor it will append through is the
    only thing that names the file it holds. Reopening the pathname to read it
    can land on a file that replaced it in the meantime, and offsets read from
    one file then truncate the other.
    """
    size = os.fstat(fd).st_size
    if size == 0:
        return RecoveryScan(file_size=0, complete_offset=0, last_sequence=0, last_event=None, had_valid_events=False)
    window = min(size, _INITIAL_TAIL_BYTES)
    while True:
        start = size - window
        data = _read_at(fd, start, window)
        newline_at = data.rfind(b"\n")
        if newline_at < 0:
            if start == 0:
                # No newline anywhere: the whole file is one torn line.
                return RecoveryScan(
                    file_size=size, complete_offset=0, last_sequence=0, last_event=None, had_valid_events=False
                )
            window = min(size, window * 2)
            continue
        complete_offset = start + newline_at + 1
        tail = _scan_tail(data[: newline_at + 1], whole_file=start == 0)
        if tail.newer_schema_version is not None:
            # Nothing past this line can be read here, its slot included,
            # so there is no sequence to resume from — only a range this
            # build would write over.
            return RecoveryScan(
                file_size=size,
                complete_offset=complete_offset,
                last_sequence=0,
                last_event=None,
                had_valid_events=False,
                newer_schema_version=tail.newer_schema_version,
            )
        if tail.event is not None:
            last_sequence = max(tail.event.sequence, tail.claimed_sequence)
            return RecoveryScan(
                file_size=size,
                complete_offset=complete_offset,
                last_sequence=last_sequence,
                last_event=tail.event,
                had_valid_events=True,
                unreadable_slots=_range_after(tail.event.sequence, last_sequence),
                unreadable_tail=tail.unreadable_tail,
            )
        if start == 0:
            # The whole file walked and nothing in it decodes. Damaged lines
            # that still name their slot took those slots — and every slot
            # below the highest of them is just as unreadable, because no
            # line anywhere in this file could be read.
            return RecoveryScan(
                file_size=size,
                complete_offset=complete_offset,
                last_sequence=tail.claimed_sequence,
                last_event=None,
                had_valid_events=False,
                unreadable_slots=_range_after(0, tail.claimed_sequence),
                unreadable_tail=tail.unreadable_tail,
            )
        window = min(size, window * 2)


def _range_after(last_readable: int, last_claimed: int) -> tuple[int, int] | None:
    """The slots between the last readable event and the last claimed one, if any."""
    if last_claimed <= last_readable:
        return None
    return last_readable + 1, last_claimed


def _read_at(fd: int, offset: int, length: int) -> bytes:
    """Read *length* bytes from *offset* without disturbing where writes land.

    ``pread`` where the platform has it; elsewhere a seek, which an append
    descriptor ignores when it writes. Either way the read is repeated until
    it has what it asked for or the file ends, because one read call is
    allowed to return less.
    """
    pread = getattr(os, "pread", None)
    chunks: list[bytes] = []
    read_so_far = 0
    while read_so_far < length:
        want = length - read_so_far
        at = offset + read_so_far
        if pread is not None:
            chunk = pread(fd, want, at)
        else:
            os.lseek(fd, at, os.SEEK_SET)
            chunk = os.read(fd, want)
        if not chunk:
            break
        chunks.append(chunk)
        read_so_far += len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class _TailScan:
    """What the last complete lines say about where a writer may resume."""

    event: TrajectoryEvent | None = None
    newer_schema_version: int | None = None
    claimed_sequence: int = 0
    """Highest slot a complete line claims, whether or not this build can read the rest of it."""
    unreadable_tail: bool = False
    """Set when a complete line past the returned event claims no slot at all."""


def _scan_tail(data: bytes, *, whole_file: bool) -> _TailScan:
    """The last decodable event among complete lines in *data*, and the slots they claim.

    Walked from the end. A line written under a newer schema stops the walk:
    a build that cannot read a line cannot know which slot it took, and
    resuming from an older event would put this runtime's events on slots the
    newer one already used. A line this build *should* have been able to read
    but cannot is damage rather than skew — the walk goes on past it — yet its
    header may still name the slot it took, and a slot named is a slot spent.
    When *data* is a tail window rather than the whole file, its first line
    may be a fragment and is skipped.
    """
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    first_index = 0 if whole_file else 1
    claimed = 0
    unreadable_tail = False
    for index in range(len(lines) - 1, first_index - 1, -1):
        line = lines[index]
        header = peek_envelope_header(line)
        if header is not None:
            if header.schema_version > SCHEMA_VERSION:
                return _TailScan(newer_schema_version=header.schema_version)
            claimed = max(claimed, header.sequence)
        try:
            return _TailScan(event=decode_event_line(line), claimed_sequence=claimed, unreadable_tail=unreadable_tail)
        except EnvelopeError:
            # A line whose header parsed spent a slot this build can name; one
            # whose header did not spent a slot nobody can name, and the walk
            # cannot tell how far past the next event the file already went.
            unreadable_tail = unreadable_tail or header is None
            continue
    return _TailScan(claimed_sequence=claimed, unreadable_tail=unreadable_tail)


def truncate_torn_tail(fd: int, scan: RecoveryScan) -> int:
    """Truncate the torn tail described by *scan* through *fd*; return bytes removed."""
    if scan.truncated_bytes <= 0:
        return 0
    os.ftruncate(fd, scan.complete_offset)
    return scan.truncated_bytes

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read-only access to a trajectory events file with corruption diagnostics.

The reader never writes back. A torn trailing line is reported (and handled
by the writer's recovery path); a bad line in the *middle* of the file is
corruption and yields an in-memory diagnostic gap rather than being skipped
silently. Unknown event types — and events written under a schema this build
does not know — are decoded, counted, and reported so that the region they
fall in is never declared exact by downstream analysis.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chrys.foundation.trajectory.envelope import (
    SCHEMA_VERSION,
    EnvelopeError,
    TrajectoryEvent,
    decode_event_line,
    iter_complete_lines,
    peek_envelope_header,
)
from chrys.foundation.trajectory.event_types import KNOWN_EVENT_TYPES, EventType


@dataclass(frozen=True, slots=True)
class CorruptLine:
    """One undecodable complete line in the middle of the file."""

    line_number: int
    byte_offset: int
    byte_length: int
    reason: str
    previous_sequence: int
    """Sequence of the last valid event before this line (0 when none)."""


@dataclass(frozen=True, slots=True)
class UnsupportedLine:
    """A complete line written under a schema this build was never meant to read."""

    line_number: int
    byte_offset: int
    sequence: int
    schema_version: int


@dataclass(frozen=True, slots=True)
class ReaderGap:
    """A sequence range the reader cannot account for (diagnostic, in-memory)."""

    after_sequence: int
    before_sequence: int | None
    reason: str


@dataclass(slots=True)
class TrajectoryReadResult:
    """Decoded events plus everything the reader could not vouch for."""

    events: list[TrajectoryEvent] = field(default_factory=list)
    corrupt_lines: list[CorruptLine] = field(default_factory=list)
    torn_tail_bytes: int = 0
    unsupported_event_count: int = 0
    """Events this build cannot vouch for: an unknown type, or a newer schema."""
    unsupported_event_sequences: list[int] = field(default_factory=list)
    unsupported_lines: list[UnsupportedLine] = field(default_factory=list)
    """Newer-schema lines kept out of *events*: their slots are real, their content is not readable here."""
    slots: list[TrajectoryEvent | UnsupportedLine] = field(default_factory=list)
    """Every line the file kept, decoded or not, in file order — what the prefix invariant is checked over."""

    @property
    def corruption_gaps(self) -> list[ReaderGap]:
        """Diagnostic gaps implied by corrupt lines, positioned between neighbours."""
        gaps: list[ReaderGap] = []
        for corrupt in self.corrupt_lines:
            following = [e.sequence for e in self.events if e.sequence > corrupt.previous_sequence]
            gaps.append(
                ReaderGap(
                    after_sequence=corrupt.previous_sequence,
                    before_sequence=min(following) if following else None,
                    reason=corrupt.reason,
                )
            )
        return gaps


def iter_file_lines(path: Path, *, chunk_size: int = 1 << 16) -> Iterator[tuple[int, bytes, bool]]:
    """Yield ``(byte_offset, line_without_newline, complete)`` for each line.

    The final tuple has ``complete=False`` only when the file ends without a
    newline (a torn tail); its bytes are the partial line.
    """
    offset = 0
    pending = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            lines, pending = iter_complete_lines(pending + chunk)
            for line in lines:
                yield offset, line, True
                offset += len(line) + 1
    if pending:
        yield offset, pending, False


def read_trajectory(
    path: Path,
    *,
    known_event_types: frozenset[str] = KNOWN_EVENT_TYPES,
) -> TrajectoryReadResult:
    """Decode every line of *path* into a :class:`TrajectoryReadResult`."""
    result = TrajectoryReadResult()
    if not path.is_file():
        return result
    line_number = 0
    previous_sequence = 0
    for offset, raw, complete in iter_file_lines(path):
        if not complete:
            result.torn_tail_bytes = len(raw)
            break
        line_number += 1
        try:
            event = decode_event_line(raw)
        except EnvelopeError as exc:
            newer = _newer_schema_line(raw, line_number=line_number, byte_offset=offset)
            if newer is not None:
                # A schema is bumped exactly when lines stop meaning what this
                # build reads them as, so failing this build's shape is what a
                # newer line is supposed to do — it is not damage. The slot is
                # real and stays accounted for; only the content is out of
                # reach.
                result.unsupported_lines.append(newer)
                result.slots.append(newer)
                result.unsupported_event_count += 1
                result.unsupported_event_sequences.append(newer.sequence)
                previous_sequence = newer.sequence
                continue
            result.corrupt_lines.append(
                CorruptLine(
                    line_number=line_number,
                    byte_offset=offset,
                    byte_length=len(raw) + 1,
                    reason=str(exc),
                    previous_sequence=previous_sequence,
                )
            )
            continue
        result.events.append(event)
        result.slots.append(event)
        previous_sequence = event.sequence
        if event.schema_version > SCHEMA_VERSION or event.event_type not in known_event_types:
            # A newer line that still fits this build's shape is reported the
            # same way an unknown type is: kept, accounted for, and named, so
            # nothing downstream calls the region it sits in exact.
            result.unsupported_event_count += 1
            result.unsupported_event_sequences.append(event.sequence)
    return result


def verify_accounted_prefix(records: Iterable[TrajectoryEvent | UnsupportedLine]) -> list[str]:
    """Check the accounted-prefix invariant over *records* in file order.

    Physical sequences must be strictly increasing with no duplicates; every
    hole must be covered by a ``trajectory.gap`` persisted **before** the first
    event after the hole. Returns human-readable violations (empty = holds).

    A line this build could not decode still took a slot on the disk it was
    written to, and it took that slot *somewhere* — so pass
    :attr:`TrajectoryReadResult.slots` to hold the file itself to the
    invariant; ``events`` alone answers only for the decoded stream.
    """
    violations: list[str] = []
    covered = CoveredSequenceRanges()
    # The prefix starts at 1, so the file's own first line is held to the same
    # rule as every later one: a log that opens at 2 is missing its head unless
    # a gap on that first line says where it went.
    previous = 0
    for record in records:
        sequence = record.sequence
        if isinstance(record, TrajectoryEvent) and record.event_type == EventType.GAP:
            first = record.payload.get("first_sequence")
            last = record.payload.get("last_sequence")
            if isinstance(first, int) and isinstance(last, int) and first <= last:
                covered.add(first, last)
        if sequence <= previous:
            violations.append(f"sequence {sequence} does not increase after {previous}")
        elif sequence > previous + 1:
            hole = covered.first_uncovered(previous + 1, sequence - 1)
            if hole is not None:
                violations.append(f"sequence {hole} is missing and not covered by an earlier gap")
        previous = sequence
    return violations


def _newer_schema_line(raw: bytes, *, line_number: int, byte_offset: int) -> UnsupportedLine | None:
    header = peek_envelope_header(raw)
    if header is None or header.schema_version <= SCHEMA_VERSION:
        return None
    return UnsupportedLine(
        line_number=line_number,
        byte_offset=byte_offset,
        sequence=header.sequence,
        schema_version=header.schema_version,
    )


class CoveredSequenceRanges:
    """Union of gap-covered sequence ranges, kept sorted, disjoint and merged.

    A log can carry a gap per event that failed to encode, so a long unhealthy
    session accumulates thousands of them — and every hole consults the whole
    accumulated history, which is quadratic when each lookup re-sorts the list.
    Kept merged, both recording a range and finding a hole bisect instead.
    """

    __slots__ = ("_ranges",)

    def __init__(self) -> None:
        self._ranges: list[tuple[int, int]] = []

    def add(self, first: int, last: int) -> None:
        """Record that a gap accounts for every sequence in ``[first, last]``."""
        if last < first:
            return
        ranges = self._ranges
        # Every stored range that overlaps or touches the new one merges with
        # it; over integers, "touches" reaches one past either end.
        lo = bisect_left(ranges, first - 1, key=_range_end)
        hi = bisect_left(ranges, (last + 2,))
        if lo < hi:
            first = min(first, ranges[lo][0])
            last = max(last, ranges[hi - 1][1])
        ranges[lo:hi] = [(first, last)]

    def first_uncovered(self, first: int, last: int) -> int | None:
        """The lowest sequence in ``[first, last]`` no gap accounts for, if any.

        Checked as intervals rather than one sequence at a time: a file
        claiming a jump of billions is corrupt, not a reason to count to a
        billion.
        """
        if last < first:
            return None
        ranges = self._ranges
        index = bisect_left(ranges, (first + 1,)) - 1
        if index < 0 or ranges[index][1] < first:
            return first
        # Merged ranges are disjoint with a break between neighbours, so the
        # first sequence past the covering range is uncovered.
        cursor = ranges[index][1] + 1
        return cursor if cursor <= last else None


def _range_end(covered_range: tuple[int, int]) -> int:
    return covered_range[1]


def last_event_of_type(events: Sequence[TrajectoryEvent], event_type: str) -> TrajectoryEvent | None:
    """Return the last event of *event_type* in *events*, if any."""
    for event in reversed(events):
        if event.event_type == event_type:
            return event
    return None

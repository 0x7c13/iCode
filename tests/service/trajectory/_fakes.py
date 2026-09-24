# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-memory trajectory sink and context builders shared by the trace helper tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext, main_actor
from chrys.foundation.trajectory.envelope import (
    LINE_BUDGET_BYTES,
    EventDraft,
    build_event,
    check_int64_range,
    encode_event_line,
    malformed_id_pointers,
)
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.writer import EmitResult
from tests.support.trajectory_invariants import assert_trajectory_operation_settlement

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
FINGERPRINT_KEY = b"k" * 32
_WRITER_ID = "0" * 32


class FakeSink:
    """Records every draft it is handed; ``fail_next`` makes the next emit raise once.

    Recording is not the whole story: the real sink hands the draft to the
    writer, which refuses a line no reader could decode and turns it into a
    gap instead. Every recorded draft therefore goes through the same checks
    here, so a trace helper that mints an ID of the wrong shape (or writes an
    over-budget line) fails in the test that emits it rather than silently
    producing a gap in production.
    """

    def __init__(self, *, fingerprint_key: bytes | None = FINGERPRINT_KEY) -> None:
        self.drafts: list[EventDraft] = []
        self.fail_next = False
        self._fingerprint_key = fingerprint_key

    @property
    def fingerprint_key(self) -> bytes | None:
        return self._fingerprint_key

    def _record(self, draft: EventDraft, payload_factory: Callable[[int], Mapping[str, Any]] | None) -> EmitResult:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("sink failure injected by test")
        sequence = len(self.drafts) + 1
        if payload_factory is not None:
            draft = replace(draft, payload=dict(payload_factory(sequence)))
        _assert_writable(draft, sequence=sequence)
        self.drafts.append(draft)
        return EmitResult.WRITTEN

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        return self._record(draft, payload_factory)

    def emit_blocking(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        return self._record(draft, payload_factory)

    def emit_soon(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult | None:
        self._record(draft, payload_factory)
        # Recorded, so it took its slot: only a sink that refuses a line
        # answers with a result here.
        return None

    # ------------------------------------------------------------ inspection

    def of_type(self, event_type: str) -> list[EventDraft]:
        return [draft for draft in self.drafts if draft.event_type == event_type]

    def only(self, event_type: str) -> EventDraft:
        matches = self.of_type(event_type)
        assert len(matches) == 1, f"expected exactly one {event_type}, got {len(matches)}"
        return matches[0]

    @property
    def event_types(self) -> list[str]:
        return [draft.event_type for draft in self.drafts]

    def assert_operations_settled(self, *, since: int = 0) -> None:
        """Apply the operation oracle to one complete recording scope."""
        assert_trajectory_operation_settlement(self.drafts[since:])


def _assert_writable(draft: EventDraft, *, sequence: int) -> None:
    """Fail the test if the writer would refuse this draft."""
    event = build_event(
        draft,
        sequence=sequence,
        runtime_id=_WRITER_ID,
        coverage_id=_WRITER_ID,
        session_id=SESSION_ID,
        branch_id=_WRITER_ID,
    )
    data = event.to_dict()
    assert not check_int64_range(data), f"{draft.event_type}: value out of int64 range"
    malformed = malformed_id_pointers(data)
    assert not malformed, f"{draft.event_type}: malformed identifiers at {malformed}"
    line = encode_event_line(event)
    assert len(line) <= LINE_BUDGET_BYTES, f"{draft.event_type}: line is {len(line)} bytes, over the writer budget"


class CancelAckSink(FakeSink):
    """A sink whose *at*-th emit records the draft, then cancels the caller.

    Models the real writer: the line itself is shielded from the producer's
    cancellation and lands regardless, so only the wait for its ack unwinds.
    """

    def __init__(self, *, at: int, fingerprint_key: bytes | None = FINGERPRINT_KEY) -> None:
        super().__init__(fingerprint_key=fingerprint_key)
        self._cancel_at = at

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        result = self._record(draft, payload_factory)
        if len(self.drafts) == self._cancel_at:
            raise asyncio.CancelledError
        return result


def make_context(
    sink: FakeSink | None = None,
    *,
    turn_id: str | None = None,
    run_operation_id: str | None = None,
) -> TrajectoryContext:
    """A main-actor context with a turn and a run, the shape the engine binds for a model run."""
    return TrajectoryContext(
        sink=sink if sink is not None else FakeSink(),
        session_id=SESSION_ID,
        actor=main_actor(SESSION_ID),
        turn_id=turn_id or new_analytics_id(),
        run_operation_id=run_operation_id or new_analytics_id(),
    )

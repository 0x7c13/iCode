# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Red/green pins for the trajectory invariant oracles themselves."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.envelope import EventDraft, TrajectoryEvent, build_event
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.reader import TrajectoryReadResult
from tests.support.trajectory_invariants import (
    assert_trajectory_accounted,
    assert_trajectory_operation_settlement,
)

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
WRITER_ID = "0" * 32


def _event(draft: EventDraft, *, sequence: int) -> TrajectoryEvent:
    return build_event(
        draft,
        sequence=sequence,
        runtime_id=WRITER_ID,
        coverage_id=WRITER_ID,
        session_id=SESSION_ID,
        branch_id=WRITER_ID,
    )


def test_operation_oracle_accepts_one_ordered_pair() -> None:
    operation_id = new_analytics_id()
    assert_trajectory_operation_settlement(
        [
            EventDraft(event_type=EventType.MODEL_RUN_STARTED, operation_id=operation_id),
            EventDraft(event_type=EventType.MODEL_RUN_FINISHED, operation_id=operation_id),
        ]
    )


def test_operation_oracle_rejects_an_unsettled_opening() -> None:
    operation_id = new_analytics_id()
    with pytest.raises(AssertionError, match="unsettled trajectory operations"):
        assert_trajectory_operation_settlement(
            [EventDraft(event_type=EventType.TOOL_OPERATION_STARTED, operation_id=operation_id)]
        )


def test_operation_oracle_rejects_a_terminal_without_an_opening() -> None:
    operation_id = new_analytics_id()
    with pytest.raises(AssertionError, match="without an earlier opening"):
        assert_trajectory_operation_settlement(
            [EventDraft(event_type=EventType.WAIT_FINISHED, operation_id=operation_id)]
        )


def test_operation_oracle_rejects_a_minted_tool_operation_with_no_events() -> None:
    operation_id = new_analytics_id()
    with pytest.raises(AssertionError, match="minted on landed calls but never opened"):
        assert_trajectory_operation_settlement([], expected_tool_operation_ids=[operation_id])


def test_accounted_prefix_oracle_accepts_an_earlier_gap() -> None:
    first = _event(EventDraft(event_type=EventType.RUNTIME_STARTED), sequence=1)
    gap = _event(
        EventDraft(
            event_type=EventType.GAP,
            payload={"first_sequence": 2, "last_sequence": 2, "dropped_count": 1, "reason": "test"},
        ),
        sequence=3,
    )
    assert_trajectory_accounted(TrajectoryReadResult(events=[first, gap], slots=[first, gap]))


def test_accounted_prefix_oracle_rejects_an_unexplained_slot() -> None:
    first = _event(EventDraft(event_type=EventType.RUNTIME_STARTED), sequence=1)
    third = _event(EventDraft(event_type=EventType.RUNTIME_FINISHED), sequence=3)
    with pytest.raises(AssertionError, match="sequence 2"):
        assert_trajectory_accounted(TrajectoryReadResult(events=[first, third], slots=[first, third]))

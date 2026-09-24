# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-cutting invariant oracles for trajectory tests.

The operation oracle is applied to complete recording scopes, not every
``FakeSink``: payload-unit tests deliberately emit only one side of a pair.
The accounted-prefix oracle always reads physical ``slots`` so an unreadable
line cannot disappear merely because it is absent from decoded ``events``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from chrys.foundation.trajectory.envelope import EventDraft, TrajectoryEvent
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.reader import TrajectoryReadResult, read_trajectory, verify_accounted_prefix

__all__ = [
    "assert_trajectory_accounted",
    "assert_trajectory_file_accounted",
    "assert_trajectory_operation_settlement",
]


class _LifecycleRecord(Protocol):
    event_type: str
    operation_id: str | None
    payload: Mapping[str, object]


def _operation_id(record: _LifecycleRecord) -> str | None:
    return record.operation_id


def _approval_request_id(record: _LifecycleRecord) -> str | None:
    request_id = record.payload.get("approval_request_id")
    return request_id if isinstance(request_id, str) and request_id else None


_LifecycleKey = Callable[[_LifecycleRecord], str | None]
_OPERATION_LIFECYCLES: tuple[tuple[str, str, str, _LifecycleKey], ...] = (
    ("workflow.run", EventType.WORKFLOW_RUN_STARTED, EventType.WORKFLOW_RUN_FINISHED, _operation_id),
    ("workflow.node", EventType.WORKFLOW_NODE_STARTED, EventType.WORKFLOW_NODE_FINISHED, _operation_id),
    ("model.run", EventType.MODEL_RUN_STARTED, EventType.MODEL_RUN_FINISHED, _operation_id),
    ("model.cycle", EventType.MODEL_CYCLE_STARTED, EventType.MODEL_CYCLE_FINISHED, _operation_id),
    ("model.exchange", EventType.MODEL_EXCHANGE_STARTED, EventType.MODEL_EXCHANGE_FINISHED, _operation_id),
    ("tool.operation", EventType.TOOL_OPERATION_STARTED, EventType.TOOL_OPERATION_FINISHED, _operation_id),
    ("approval", EventType.APPROVAL_REQUESTED, EventType.APPROVAL_RESOLVED, _approval_request_id),
    ("compaction", EventType.COMPACTION_STARTED, EventType.COMPACTION_FINISHED, _operation_id),
    ("sub_agent", EventType.SUB_AGENT_STARTED, EventType.SUB_AGENT_FINISHED, _operation_id),
    ("wait", EventType.WAIT_STARTED, EventType.WAIT_FINISHED, _operation_id),
    ("hook.operation", EventType.HOOK_OPERATION_STARTED, EventType.HOOK_OPERATION_FINISHED, _operation_id),
    ("preparation", EventType.PREPARATION_STARTED, EventType.PREPARATION_FINISHED, _operation_id),
)


def assert_trajectory_operation_settlement(
    records: Iterable[EventDraft | TrajectoryEvent],
    *,
    expected_tool_operation_ids: Iterable[str] = (),
) -> None:
    """Assert that every operation opening has exactly one later terminal.

    The check is ordered and family-aware: reusing an id across unrelated
    lifecycle types cannot accidentally settle it, a terminal before its
    opening is rejected, and duplicate openings/terminals are rejected.
    ``expected_tool_operation_ids`` links the event oracle back to landed call
    metadata, catching an id minted without even a started event.
    """
    lifecycle_by_event: dict[str, tuple[str, bool, _LifecycleKey]] = {}
    for family, started, finished, key in _OPERATION_LIFECYCLES:
        lifecycle_by_event[started] = (family, True, key)
        lifecycle_by_event[finished] = (family, False, key)

    open_operations: dict[tuple[str, str], int] = {}
    opened_tool_operation_ids: set[str] = set()
    for index, record in enumerate(records):
        lifecycle = lifecycle_by_event.get(record.event_type)
        if lifecycle is None:
            continue
        family, is_start, key_reader = lifecycle
        key = key_reader(record)
        assert key is not None, f"{record.event_type} at index {index} has no lifecycle key"
        identity = (family, key)
        if is_start:
            assert identity not in open_operations, (
                f"{family} operation {key} opens twice at indexes {open_operations.get(identity)} and {index}"
            )
            open_operations[identity] = index
            if family == "tool.operation":
                opened_tool_operation_ids.add(key)
        else:
            assert identity in open_operations, (
                f"{family} operation {key} closes at index {index} without an earlier opening"
            )
            del open_operations[identity]

    assert not open_operations, "unsettled trajectory operations: " + ", ".join(
        f"{family} {key} (opened at index {index})" for (family, key), index in sorted(open_operations.items())
    )
    missing_tool_operations = set(expected_tool_operation_ids) - opened_tool_operation_ids
    assert not missing_tool_operations, "tool operation ids minted on landed calls but never opened: " + ", ".join(
        sorted(missing_tool_operations)
    )


def assert_trajectory_accounted(result: TrajectoryReadResult) -> None:
    """Assert the on-disk accounted-prefix invariant over physical slots."""
    violations = verify_accounted_prefix(result.slots)
    assert violations == [], "trajectory prefix is not fully accounted:\n" + "\n".join(violations)


def assert_trajectory_file_accounted(path: Path) -> TrajectoryReadResult:
    """Read *path*, assert its physical prefix, and return the same result."""
    result = read_trajectory(path)
    assert_trajectory_accounted(result)
    return result

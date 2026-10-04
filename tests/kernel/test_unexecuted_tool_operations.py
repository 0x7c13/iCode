# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Trajectory records for calls the tool pipeline never runs.

An unknown tool or a call whose arguments fail pre-pipeline validation never
reaches the middleware, so the kernel is the only place that can close its
tool operation — and the only place that knows the provenance landing
stamped on the call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG, trajectory_scope
from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.event_types import EventType, RuntimeFinishReason, ToolOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import OPERATION_ID_KEY
from chrys.foundation.trajectory.reader import read_trajectory
from chrys.foundation.trajectory.writer import EmitResult
from chrys.kernel import FunctionTool
from chrys.kernel._content import Content
from chrys.kernel._tool_execution import _record_unexecuted_tool_operation
from chrys.kernel.types import ChatResponse, ChatResponseUpdate, Message
from chrys.service.trajectory.session import SessionTrajectory, trajectory_events_path
from tests.kernel._fakes import _final_response, _stack, _text_response, _text_update, _user
from tests.service.trajectory._fakes import SESSION_ID, CancelAckSink, FakeSink, make_context
from tests.support.trajectory_invariants import assert_trajectory_accounted


def _unknown_call(**properties: object) -> Content:
    call = Content.from_function_call(call_id="c1", name="search_issues", arguments={"query": "open"})
    call.additional_properties[OPERATION_ID_KEY] = new_analytics_id()
    call.additional_properties.update(properties)
    return call


def _error_result() -> Content:
    return Content.from_function_result(call_id="c1", result="Error: tool not found")


@pytest.mark.asyncio
async def test_a_call_that_never_ran_still_names_the_server_behind_it() -> None:
    sink = FakeSink()
    call = _unknown_call(**{TOOL_CALL_CONTEXT_METADATA_KEY: {"server_name": "github"}})

    with trajectory_scope(make_context(sink)):
        await _record_unexecuted_tool_operation(call, outcome=ToolOutcome.UNKNOWN_TOOL, result=_error_result())

    assert sink.only(EventType.TOOL_OPERATION_STARTED).payload["tool_context"] == {"server_name": "github"}
    assert sink.only(EventType.TOOL_OPERATION_FINISHED).payload["outcome"] == ToolOutcome.UNKNOWN_TOOL


@pytest.mark.asyncio
async def test_a_call_with_no_stamped_provenance_omits_the_field() -> None:
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        await _record_unexecuted_tool_operation(
            _unknown_call(), outcome=ToolOutcome.INVALID_ARGUMENTS, result=_error_result()
        )

    assert "tool_context" not in sink.only(EventType.TOOL_OPERATION_STARTED).payload


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_call_whose_arguments_are_not_an_object_closes_as_invalid_arguments(stream: bool) -> None:
    """The loop refuses a non-object payload before the pipeline, so it closes the operation itself."""
    sink = FakeSink()
    ran: list[dict[str, Any]] = []

    async def lenient(**kwargs: Any) -> str:
        ran.append(kwargs)
        return "ran"

    tool = FunctionTool(
        name="lenient", func=lenient, input_model={"type": "object", "properties": {"state": {"type": "string"}}}
    )
    call = Content.from_function_call("c1", "lenient", arguments="[1]")
    if stream:
        turns: list[Any] = [[ChatResponseUpdate(contents=[call], role="assistant")], [_text_update("done")]]
    else:
        turns = [ChatResponse(messages=[Message("assistant", [call])]), _text_response()]
    layer, _wire = _stack(turns)

    await _final_response(
        layer,
        [_user()],
        stream=stream,
        options={"tools": [tool]},
        client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
    )

    assert ran == []
    started = sink.only(EventType.TOOL_OPERATION_STARTED)
    finished = sink.only(EventType.TOOL_OPERATION_FINISHED)
    assert finished.operation_id == started.operation_id
    assert finished.payload["outcome"] == ToolOutcome.INVALID_ARGUMENTS


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.asyncio
async def test_a_call_whose_opening_line_was_refused_is_not_closed_either(tmp_path: Path, queued: bool) -> None:
    """An unknown tool's name is the model's: one past the line budget makes the
    opening event unwritable, and a terminal behind that gap would close an
    operation the log never opened — on the awaited path and on the cancelled
    one, which learns the same answer from the sequence it never got."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    trajectory = SessionTrajectory(
        session_id=SESSION_ID,
        session_dir=session_dir,
        config_dir=tmp_path / "config",
        persisted_state_probe=lambda _path: False,
    )
    assert await trajectory.ensure_active() is True
    call = Content.from_function_call(call_id="c1", name="x" * 5000, arguments={})
    call.additional_properties[OPERATION_ID_KEY] = new_analytics_id()

    with trajectory_scope(trajectory.context()):
        await _record_unexecuted_tool_operation(
            call, outcome=ToolOutcome.UNKNOWN_TOOL, result=_error_result(), queued=queued
        )
    assert await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN) is True

    result = read_trajectory(trajectory_events_path(session_dir))
    types = [event.event_type for event in result.events]
    assert EventType.TOOL_OPERATION_STARTED not in types
    assert EventType.TOOL_OPERATION_FINISHED not in types
    # The slot the refused line took is still accounted for.
    assert EventType.GAP in types
    assert_trajectory_accounted(result)


class _NeverAcknowledges(FakeSink):
    """Records every line, then never answers the ack the writer owes it."""

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        self._record(draft, payload_factory)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_settling_a_cancelled_batch_does_not_wait_for_the_writer() -> None:
    """The unwind runs where a Stop is being served: both lines take their
    sequence here and are acknowledged in the background, so a writer that is
    slow to answer cannot hold the interrupt behind it."""
    sink = _NeverAcknowledges()

    with trajectory_scope(make_context(sink)):
        await asyncio.wait_for(
            _record_unexecuted_tool_operation(_unknown_call(), outcome=ToolOutcome.FILTERED, queued=True),
            timeout=5,
        )

    assert sink.event_types == [EventType.TOOL_OPERATION_STARTED, EventType.TOOL_OPERATION_FINISHED]


@pytest.mark.asyncio
async def test_a_call_interrupted_between_the_pair_still_closes_its_operation() -> None:
    """The start marker's line lands even when its ack wait is cancelled, so
    the terminal one has to follow it."""
    sink = CancelAckSink(at=1)

    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await _record_unexecuted_tool_operation(
            _unknown_call(), outcome=ToolOutcome.UNKNOWN_TOOL, result=_error_result()
        )

    assert sink.only(EventType.TOOL_OPERATION_STARTED)
    assert sink.only(EventType.TOOL_OPERATION_FINISHED).payload["outcome"] == ToolOutcome.UNKNOWN_TOOL


@pytest.mark.asyncio
async def test_a_terminal_interrupted_on_its_own_ack_is_not_written_twice() -> None:
    """The terminal's line is committed before its ack can unwind, so the
    rescue behind the start marker must not queue a second copy."""
    sink = CancelAckSink(at=2)

    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await _record_unexecuted_tool_operation(
            _unknown_call(), outcome=ToolOutcome.UNKNOWN_TOOL, result=_error_result()
        )

    assert sink.event_types == [EventType.TOOL_OPERATION_STARTED, EventType.TOOL_OPERATION_FINISHED]

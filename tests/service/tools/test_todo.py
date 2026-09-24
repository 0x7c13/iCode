# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for todo_write tool and TodoMiddleware."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import TodoListUpdated
from chrys.foundation.models.todos import MAX_TODO_ITEMS, MAX_TODO_TEXT_LENGTH, TodoItem
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY, TOOL_FAILED_METADATA_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware import TodoMiddleware
from chrys.service.todos.tracker import TodoTracker
from chrys.service.tools.builtins.todo import todo_write
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedTodoClient:
    """Innermost fake wire client replaying scripted assistant turns."""

    def __init__(self, turns: list[list[Content] | str]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []

    def get_response(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        options: dict[str, object] | None = None,
        **_kwargs: object,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        assert not stream
        assert options is not None
        turn = self.turns.pop(0)
        contents: list[Any] = [turn] if isinstance(turn, str) else list(turn)
        return ChatResponse(messages=[Message(role="assistant", contents=contents)])


def _todo_call(call_id: str, todos: list[dict[str, str]]) -> Content:
    return Content.from_function_call(call_id=call_id, name="todo_write", arguments={"todos": todos})


def _function_results(messages: list[Message]) -> list[Content]:
    return [item for msg in messages for item in msg.contents if item.type == "function_result"]


def _mock_todo_context(arguments: object) -> MagicMock:
    mock_function = MagicMock()
    mock_function.chrys_kind = "todo"
    mock_function.name = "todo_write"
    context = MagicMock()
    context.function = mock_function
    context.arguments = arguments
    context.result = None
    context.same_tool_calls_in_batch = 1
    return context


def _capture_tool_metadata() -> tuple[dict[str, object], object]:
    metadata: dict[str, object] = {}
    return metadata, tool_result_metadata.set(metadata)


class _CountingTodoMiddleware(TodoMiddleware):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen = 0

    async def process(self, context: Any, call_next: Any) -> None:
        self.seen += 1
        await super().process(context, call_next)


# ---------------------------------------------------------------------------
# Tool fallback + schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_write_without_middleware_records_structured_failure() -> None:
    """The unwired fallback returns a structured tool_error, not a plain string."""
    metadata, token = _capture_tool_metadata()
    try:
        result = await todo_write([])
    finally:
        tool_result_metadata.reset(token)

    assert result == "Error: todo tracking is not available in this session."
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "todo_unavailable"


def test_todo_write_schema_expresses_status_literal_and_replacement() -> None:
    """The status Literal must live in the JSON schema — kernel pre-pipeline
    validation (not TodoMiddleware) owns type/shape/status errors."""
    schema = todo_write.parameters()
    assert schema["required"] == ["todos"]
    assert "Replaces the previous list entirely" in schema["properties"]["todos"]["description"]
    flattened = json.dumps(schema)
    for status in ("pending", "in_progress", "completed"):
        assert status in flattened


# ---------------------------------------------------------------------------
# Middleware — gating and validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_middleware_passes_through_non_todo() -> None:
    mw = TodoMiddleware(EventBus(), TodoTracker(), session_id="test")

    mock_function = MagicMock()
    mock_function.chrys_kind = "shell"
    context = MagicMock()
    context.function = mock_function

    call_next = AsyncMock()
    await mw.process(context, call_next)

    call_next.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected_fragment"),
    [
        ({"question": "no todos key"}, "requires a 'todos' list"),
        ({"todos": "not a list"}, "requires a 'todos' list"),
        (
            {"todos": [{"content": f"item {i}"} for i in range(MAX_TODO_ITEMS + 1)]},
            f"too many todo items ({MAX_TODO_ITEMS + 1})",
        ),
        ({"todos": [{"content": "x" * (MAX_TODO_TEXT_LENGTH + 1)}]}, "content exceeds"),
        (
            {"todos": [{"content": "ok", "active_form": "y" * (MAX_TODO_TEXT_LENGTH + 1)}]},
            "active_form exceeds",
        ),
        ({"todos": [{"content": "   "}]}, "empty content"),
    ],
)
async def test_todo_middleware_validation_rejects_without_mutating_tracker(
    arguments: dict[str, object], expected_fragment: str
) -> None:
    bus = EventBus()
    tracker = TodoTracker()
    sentinel = (TodoItem(content="pre-existing"),)
    await tracker.replace(sentinel)
    published: list[TodoListUpdated] = []

    async def collect(event: TodoListUpdated) -> None:
        published.append(event)

    await bus.subscribe(TodoListUpdated, collect)
    mw = TodoMiddleware(bus, tracker, session_id="test")
    context = _mock_todo_context(arguments)
    call_next = AsyncMock()

    metadata, token = _capture_tool_metadata()
    try:
        await mw.process(context, call_next)
    finally:
        tool_result_metadata.reset(token)

    assert context.result.startswith("Error:")
    assert expected_fragment in context.result
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "todo_invalid"
    assert tracker.snapshot() == sentinel
    assert published == []
    call_next.assert_not_called()


@pytest.mark.asyncio
async def test_todo_middleware_batch_conflict_rejects_all_duplicates() -> None:
    bus = EventBus()
    tracker = TodoTracker()
    sentinel = (TodoItem(content="pre-existing"),)
    await tracker.replace(sentinel)
    mw = TodoMiddleware(bus, tracker, session_id="test")

    context = _mock_todo_context({"todos": [{"content": "would win"}]})
    context.same_tool_calls_in_batch = 2
    call_next = AsyncMock()

    metadata, token = _capture_tool_metadata()
    try:
        await mw.process(context, call_next)
    finally:
        tool_result_metadata.reset(token)

    assert "multiple todo_write calls in one response" in context.result
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "todo_batch_conflict"
    assert tracker.snapshot() == sentinel
    call_next.assert_not_called()


# ---------------------------------------------------------------------------
# Middleware — apply + publish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_middleware_applies_and_publishes_after_lock_release() -> None:
    """The event carries the applied list; publish happens after the tracker
    lock is released — a subscriber that awaits a tracker WRITER completes
    instead of deadlocking."""
    bus = EventBus()
    tracker = TodoTracker()
    mw = TodoMiddleware(bus, tracker, session_id="sess-1")
    published: list[TodoListUpdated] = []

    async def rewrite_from_event(event: TodoListUpdated) -> None:
        published.append(event)
        # A lock-held publish would deadlock here (EventBus.publish awaits
        # subscribers inline; replace() needs the same lock).
        await asyncio.wait_for(tracker.replace(tuple(event.items)), timeout=1)

    await bus.subscribe(TodoListUpdated, rewrite_from_event)

    todos = [
        {"content": "Fix the auth bug", "status": "completed", "active_form": "Fixing the auth bug"},
        {"content": "Run tests", "status": "in_progress", "active_form": "Running tests"},
        {"content": "Update docs"},
    ]
    context = _mock_todo_context({"todos": todos})
    call_next = AsyncMock()
    await asyncio.wait_for(mw.process(context, call_next), timeout=5)

    expected = (
        TodoItem(content="Fix the auth bug", status="completed", active_form="Fixing the auth bug"),
        TodoItem(content="Run tests", status="in_progress", active_form="Running tests"),
        TodoItem(content="Update docs"),
    )
    assert context.result == "Todo list updated: 3 items (1 completed, 1 in progress, 1 pending)."
    assert tracker.snapshot() == expected
    assert len(published) == 1
    assert tuple(published[0].items) == expected
    assert published[0].session_id == "sess-1"
    call_next.assert_not_called()


# ---------------------------------------------------------------------------
# Through the kernel tool loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_write_bad_status_fails_kernel_validation_before_middleware() -> None:
    """A bad status Literal is an argument_parsing error stamped by the KERNEL;
    TodoMiddleware never runs."""
    bus = EventBus()
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="sentinel"),))
    mw = _CountingTodoMiddleware(bus, tracker, session_id="sess-1")

    wire = _ScriptedTodoClient(
        [
            [
                Content.from_function_call(
                    call_id="c1", name="todo_write", arguments={"todos": [{"content": "x", "status": "done"}]}
                )
            ],
            "done",
        ]
    )
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [todo_write]},
        middleware=[mw],
    )

    assert response.text == "done"
    results = _function_results(wire.calls[1])
    assert len(results) == 1
    result_text = str(results[0].result)
    assert result_text.startswith("Error: Invalid arguments for 'todo_write':")
    assert "todos[0].status: expected" in result_text
    assert '"pending" | "in_progress" | "completed"' in result_text
    assert "Expected: {todos: [{content: string," in result_text
    assert len(result_text) < 300
    assert results[0].additional_properties[TOOL_FAILED_METADATA_KEY] is True
    assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
    assert mw.seen == 0
    assert tracker.snapshot() == (TodoItem(content="sentinel"),)


@pytest.mark.asyncio
async def test_todo_write_duplicate_batch_rejects_both_and_leaves_tracker_unchanged() -> None:
    """Two schema-valid todo_write calls in ONE batch → two conflict errors and
    the tracker list content is unchanged (no scheduling-defined winner)."""
    bus = EventBus()
    tracker = TodoTracker()
    sentinel = (TodoItem(content="sentinel"),)
    await tracker.replace(sentinel)
    mw = _CountingTodoMiddleware(bus, tracker, session_id="sess-1")
    published: list[TodoListUpdated] = []

    async def collect(event: TodoListUpdated) -> None:
        published.append(event)

    await bus.subscribe(TodoListUpdated, collect)

    wire = _ScriptedTodoClient(
        [
            [
                _todo_call("c1", [{"content": "first list"}]),
                _todo_call("c2", [{"content": "second list"}]),
            ],
            "done",
        ]
    )
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [todo_write]},
        middleware=[mw],
    )

    assert response.text == "done"
    results = _function_results(wire.calls[1])
    assert len(results) == 2
    for result in results:
        assert result.result == (
            "Error: multiple todo_write calls in one response; send a single call with the complete list."
        )
    assert mw.seen == 2
    assert tracker.snapshot() == sentinel
    assert published == []


@pytest.mark.asyncio
async def test_todo_write_singleton_applies_through_loop() -> None:
    bus = EventBus()
    tracker = TodoTracker()
    mw = _CountingTodoMiddleware(bus, tracker, session_id="sess-1")
    published: list[TodoListUpdated] = []

    async def collect(event: TodoListUpdated) -> None:
        published.append(event)

    await bus.subscribe(TodoListUpdated, collect)

    wire = _ScriptedTodoClient(
        [
            [
                _todo_call(
                    "c1",
                    [
                        {"content": "Write tests", "status": "in_progress", "active_form": "Writing tests"},
                        {"content": "Run gates"},
                    ],
                )
            ],
            "done",
        ]
    )
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [todo_write]},
        middleware=[mw],
    )

    assert response.text == "done"
    results = _function_results(wire.calls[1])
    assert len(results) == 1
    assert results[0].result == "Todo list updated: 2 items (0 completed, 1 in progress, 1 pending)."
    assert tracker.snapshot() == (
        TodoItem(content="Write tests", status="in_progress", active_form="Writing tests"),
        TodoItem(content="Run gates"),
    )
    assert len(published) == 1
    assert published[0].session_id == "sess-1"


@pytest.mark.asyncio
async def test_todo_write_back_to_back_singleton_batches_both_apply() -> None:
    """Two sequential single-call batches through the SAME middleware instance
    both apply — the duplicate counter is per batch, not per run."""
    bus = EventBus()
    tracker = TodoTracker()
    mw = _CountingTodoMiddleware(bus, tracker, session_id="sess-1")
    published: list[TodoListUpdated] = []

    async def collect(event: TodoListUpdated) -> None:
        published.append(event)

    await bus.subscribe(TodoListUpdated, collect)

    wire = _ScriptedTodoClient(
        [
            [_todo_call("c1", [{"content": "step one", "status": "in_progress"}])],
            [
                _todo_call(
                    "c2",
                    [{"content": "step one", "status": "completed"}, {"content": "step two", "status": "in_progress"}],
                )
            ],
            "done",
        ]
    )
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [todo_write]},
        middleware=[mw],
    )

    assert response.text == "done"
    assert mw.seen == 2
    assert tracker.snapshot() == (
        TodoItem(content="step one", status="completed"),
        TodoItem(content="step two", status="in_progress"),
    )
    assert [len(event.items) for event in published] == [1, 2]

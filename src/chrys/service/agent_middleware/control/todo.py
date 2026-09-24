# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Todo middleware — intercepts todo_write tool calls to update the session todo list."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from chrys.foundation.events.types import TodoListUpdated
from chrys.foundation.models.todos import MAX_TODO_ITEMS, MAX_TODO_TEXT_LENGTH, TodoItem, TodoStatus
from chrys.foundation.tool_kinds import KIND_TODO, get_tool_kind
from chrys.kernel.middleware import FunctionMiddleware
from chrys.service.tools.result_metadata import tool_error

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chrys.foundation.events.bus import EventBus
    from chrys.kernel.middleware import FunctionInvocationContext
    from chrys.service.todos.tracker import TodoTracker


class TodoMiddleware(FunctionMiddleware):
    """Intercepts ``todo_write`` calls: validate → apply to the tracker → publish.

    Follows the ``AskUserMiddleware`` intercept pattern — the underlying tool
    function is never invoked; ``context.result`` carries the outcome.

    Validation here covers only what the JSON schema cannot express (item
    count, per-field length caps, non-empty content). Type/shape/status errors
    fail kernel pre-pipeline argument validation and never reach this
    middleware.
    """

    def __init__(self, event_bus: EventBus, tracker: TodoTracker, session_id: str | None = None) -> None:
        self._bus = event_bus
        self._tracker = tracker
        self._session_id = session_id

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if get_tool_kind(context.function) != KIND_TODO:
            await call_next()
            return

        # Same-batch duplicates: with whole-list replacement, the surviving
        # list of two gathered todo_write calls would be scheduling-defined.
        # Reject ALL of them — zero winners is deterministic; the model
        # resends a single call with the complete list.
        if context.same_tool_calls_in_batch > 1:
            context.result = tool_error(
                "todo_batch_conflict",
                "multiple todo_write calls in one response; send a single call with the complete list.",
            )
            return

        args = context.arguments if isinstance(context.arguments, dict) else {}
        raw_todos = args.get("todos")
        if not isinstance(raw_todos, list):
            context.result = tool_error("todo_invalid", "todo_write requires a 'todos' list.")
            return
        if len(raw_todos) > MAX_TODO_ITEMS:
            context.result = tool_error(
                "todo_invalid",
                f"too many todo items ({len(raw_todos)}); the list is capped at {MAX_TODO_ITEMS}.",
            )
            return

        items: list[TodoItem] = []
        for index, entry in enumerate(raw_todos):
            data = entry if isinstance(entry, dict) else {}
            content = data.get("content", "")
            active_form = data.get("active_form", "")
            if not isinstance(content, str) or not content.strip():
                context.result = tool_error("todo_invalid", f"todo item {index + 1} has empty content.")
                return
            if len(content) > MAX_TODO_TEXT_LENGTH:
                context.result = tool_error(
                    "todo_invalid",
                    f"todo item {index + 1} content exceeds {MAX_TODO_TEXT_LENGTH} characters.",
                )
                return
            if isinstance(active_form, str) and len(active_form) > MAX_TODO_TEXT_LENGTH:
                context.result = tool_error(
                    "todo_invalid",
                    f"todo item {index + 1} active_form exceeds {MAX_TODO_TEXT_LENGTH} characters.",
                )
                return
            items.append(
                TodoItem(
                    content=content,
                    # Kernel schema validation admits only ``TodoStatus``
                    # literals before middleware dispatch.
                    status=cast("TodoStatus", data.get("status", "pending")),
                    active_form=active_form if isinstance(active_form, str) else "",
                )
            )

        applied = await self._tracker.replace(tuple(items))
        # Publish AFTER the tracker lock is released: EventBus.publish awaits
        # subscriber callbacks inline, so a lock-held publish would span
        # arbitrary UI/ACP handlers. The event carries the exact applied
        # snapshot, and reject-all above means at most one apply per batch, so
        # published state cannot invert. session_id is REQUIRED: host streams
        # drop sessionless events in _event_belongs_to_session.
        await self._bus.publish(TodoListUpdated(items=list(applied), session_id=self._session_id))

        completed = sum(1 for item in applied if item.status == "completed")
        in_progress = sum(1 for item in applied if item.status == "in_progress")
        pending = len(applied) - completed - in_progress
        context.result = (
            f"Todo list updated: {len(applied)} items "
            f"({completed} completed, {in_progress} in progress, {pending} pending)."
        )

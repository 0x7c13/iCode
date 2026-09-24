# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""todo_write tool — session todo-list maintenance.

The real behavior lives in ``TodoMiddleware`` (a ``FunctionMiddleware``),
which intercepts calls to this tool, validates the list, applies it to the
engine-owned ``TodoTracker`` and publishes ``TodoListUpdated``.  The tool
function itself is only reached as a fallback when the middleware is not
wired (e.g. a sub-agent profile that lists the ``todo`` category, or direct
invocation in tests) — unlike ``ask_user``'s plain-string fallback it returns
a structured ``tool_error`` so the failure records metadata.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from chrys.foundation.models.todos import TodoStatus
from chrys.service.tools.kinds import KIND_TODO, tool
from chrys.service.tools.result_metadata import tool_error


class TodoItemParam(BaseModel):
    """One todo entry as supplied by the model."""

    content: str
    status: TodoStatus = "pending"
    active_form: str = ""


@tool(kind=KIND_TODO)
async def todo_write(
    todos: Annotated[
        list[TodoItemParam],
        "The complete todo list. Replaces the previous list entirely.",
    ],
) -> str:
    """Use this tool to create and manage a structured task list for the current session.

    It helps you track progress and shows the user a live checklist of what
    you're doing.

    **When to use**: multi-step tasks (3+ distinct steps), work spanning
    multiple files or subsystems, or several tasks given at once.
    **When NOT to use**: single straightforward tasks or purely conversational
    turns.

    **Semantics**: each call replaces the entire list — always send the full
    updated list. Remove an item by omitting it. Call ``todo_write`` at most
    once per response — never in parallel with another ``todo_write``. Keep
    exactly ONE item ``in_progress`` at a time. Mark an item ``completed``
    immediately after finishing it; never mark it completed if tests failed or
    work is partial. ``content`` is imperative ("Fix the auth bug");
    ``active_form`` is the present-continuous label shown while in progress
    ("Fixing the auth bug").

    **Language**: write ``content`` and ``active_form`` in the language the
    user writes in (e.g. a Chinese request gets Chinese todo items); when the
    user's language is unclear or mixed, default to English.
    """
    # Fallback when TodoMiddleware is not in the pipeline (sub-agent builds, tests).
    return tool_error("todo_unavailable", "todo tracking is not available in this session.")

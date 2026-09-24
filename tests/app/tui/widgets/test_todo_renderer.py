# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the todo_write inline checklist renderer."""

from __future__ import annotations

import json

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets.chat.renderers.todo import TodoToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY

_TODOS = [
    {"content": "write tests", "status": "completed", "active_form": ""},
    {"content": "run suite", "status": "in_progress", "active_form": "Running suite"},
    {"content": "ship it", "status": "pending", "active_form": ""},
]


class TodoCardApp(App):
    def __init__(self, card: TodoToolCall) -> None:
        super().__init__()
        self._card = card

    def compose(self) -> ComposeResult:
        yield self._card


def _console(width: int) -> Console:
    # Textual's run_test stdout proxy reports itself as a terminal; force
    # plain rendering so captured output carries no ANSI codes.
    return Console(width=width, force_terminal=False, _environ={})


def _body_lines(card: TodoToolCall, width: int = 60) -> list[str]:
    content = card.query_one("#tc-body", Static).content
    console = _console(width)
    with console.capture() as capture:
        console.print(content)
    return [line.rstrip() for line in capture.get().splitlines() if line.strip()]


def _body_styles(card: TodoToolCall, width: int = 60) -> list[str]:
    content = card.query_one("#tc-body", Static).content
    return [str(segment.style) for segment in _console(width).render(content) if segment.style]


def test_todo_renderer_uses_base_tool_card_contract() -> None:
    tool = TodoToolCall("c1", "todo_write", args_summary=json.dumps({"todos": _TODOS}))

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == "todo_write"
    assert tool.status == "running"


def test_todo_renderer_parses_items_from_args_summary_fallback() -> None:
    tool = TodoToolCall("c1", "todo_write", args_summary=json.dumps({"todos": _TODOS}))
    assert [item.content for item in tool._todo_items()] == ["write tests", "run suite", "ship it"]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_todo_renderer_border_title_counts_and_checklist_body(theme: str) -> None:
    card = TodoToolCall("c1", "todo_write", args={"todos": _TODOS})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        pilot.app.theme = theme
        await pilot.pause()
        panel = card.query_one("#tc-panel")
        assert str(panel.border_title) == "Todo List (1/3)"

        card.set_complete("Todo list updated: 3 items (1 completed, 1 in progress, 1 pending).")
        await pilot.pause()

        assert card.status == "complete"
        assert card.has_class("-success")
        assert _body_lines(card) == ["✓ write tests", "▸ Running suite", "☐ ship it"]
        warning = pilot.app.theme_variables["warning"]
        expected_style = str(rich_style_from_textual_color(warning, bold=True)).lower()
        assert expected_style in [style.lower() for style in _body_styles(card)]
        # Raw result stays available for copy even though the body shows the list.
        assert card.result_text.startswith("Todo list updated")


@pytest.mark.asyncio
async def test_todo_renderer_wraps_long_items_with_hanging_indent() -> None:
    long_item = "Read orchestration: engine/run/* remaining modules (compaction, finalizer, turn_state, retry)"
    card = TodoToolCall("c1", "todo_write", args={"todos": [{"content": long_item, "status": "pending"}]})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        card.set_complete("Todo list updated: 1 item.")
        await pilot.pause()

        lines = _body_lines(card, width=40)
        assert len(lines) > 1
        assert lines[0].startswith("☐ Read orchestration:")
        # Continuation lines hang under the first content character, not the marker.
        for continuation in lines[1:]:
            assert continuation.startswith("  ")
            assert continuation[2] != " "
        assert " ".join(line[2:].strip() for line in lines) == long_item


@pytest.mark.asyncio
async def test_todo_renderer_treats_model_markup_as_literal_text() -> None:
    """Model-authored todo content full of Textual markup must render literally.

    The checklist pipeline builds Rich ``Text`` everywhere; if any step
    regresses to raw strings, rendering raises ``MarkupError`` (crashing the
    frame) or silently styles/injects ``[@click=...]`` actions.
    """
    hostile = [
        {"content": "[bold] unclosed tag", "status": "pending"},
        {"content": "stray close [/]", "status": "in_progress", "active_form": "closing [/] now"},
        {"content": "mismatched [red]x[/blue]", "status": "completed"},
        {"content": "array a[5] and [@click=app.quit]boom[/]", "status": "pending"},
    ]
    card = TodoToolCall("c1", "todo_write", args={"todos": hostile})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        card.set_complete("Todo list updated: 4 items.")
        await pilot.pause()

        assert _body_lines(card, width=70) == [
            "☐ [bold] unclosed tag",
            "▸ closing [/] now",
            "✓ mismatched [red]x[/blue]",
            "☐ array a[5] and [@click=app.quit]boom[/]",
        ]


@pytest.mark.asyncio
async def test_todo_renderer_error_keeps_text_body() -> None:
    card = TodoToolCall("c1", "todo_write", args={"todos": _TODOS})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        card.set_complete(
            "Error: todo_invalid: too many items.",
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        await pilot.pause()

        assert card.status == "error"
        assert card.has_class("-error")
        assert _body_lines(card) == ["Error: todo_invalid: too many items."]


@pytest.mark.asyncio
async def test_todo_renderer_unparseable_args_falls_back_to_result_text() -> None:
    card = TodoToolCall("c1", "todo_write", args={"todos": "not a list"})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        panel = card.query_one("#tc-panel")
        assert str(panel.border_title) == "Todo List"

        card.set_complete("Todo list updated: 0 items.")
        await pilot.pause()

        assert _body_lines(card) == ["Todo list updated: 0 items."]


@pytest.mark.asyncio
async def test_todo_renderer_update_args_refreshes_title() -> None:
    card = TodoToolCall("c1", "todo_write", args={"todos": _TODOS})
    async with TodoCardApp(card).run_test(size=(80, 24)) as pilot:
        card.update_args({"todos": [{"content": "only one", "status": "completed"}]})
        await pilot.pause()

        panel = card.query_one("#tc-panel")
        assert str(panel.border_title) == "Todo List (1/1)"

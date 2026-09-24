# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared todo-list data model and its tolerant decoder."""

from __future__ import annotations

from chrys.foundation.models.todos import (
    MAX_TODO_ITEMS,
    MAX_TODO_TEXT_LENGTH,
    TODO_STATUSES,
    TodoItem,
    parse_todo_items,
    serialize_todo_items,
)


def test_todo_item_defaults() -> None:
    item = TodoItem(content="Run tests")
    assert item.status == "pending"
    assert item.active_form == ""


def test_constants() -> None:
    assert sorted(TODO_STATUSES) == ["completed", "in_progress", "pending"]
    assert MAX_TODO_ITEMS == 128
    assert MAX_TODO_TEXT_LENGTH == 500


def test_serialize_parse_round_trip() -> None:
    items = (
        TodoItem(content="Fix the auth bug", status="completed", active_form="Fixing the auth bug"),
        TodoItem(content="Run the test suite", status="in_progress", active_form="Running tests"),
        TodoItem(content="Update docs"),
    )
    payload = serialize_todo_items(items)
    assert payload == [
        {"content": "Fix the auth bug", "status": "completed", "active_form": "Fixing the auth bug"},
        {"content": "Run the test suite", "status": "in_progress", "active_form": "Running tests"},
        {"content": "Update docs", "status": "pending", "active_form": ""},
    ]
    assert parse_todo_items(payload) == items


def test_parse_non_list_yields_empty_tuple() -> None:
    assert parse_todo_items(None) == ()
    assert parse_todo_items("not a list") == ()
    assert parse_todo_items({"content": "dict, not list"}) == ()
    assert parse_todo_items(42) == ()


def test_parse_skips_garbage_entries_but_keeps_valid_ones() -> None:
    payload = [
        "not a dict",
        {"content": ""},  # empty content
        {"content": "   "},  # whitespace-only content
        {"content": 123},  # non-string content
        {"status": "pending"},  # missing content
        {"content": "valid", "status": "sleeping"},  # unknown status
        {"content": "valid", "status": []},  # unhashable status
        {"content": "kept", "status": "in_progress", "active_form": "Keeping"},
        {"content": "kept too"},  # defaults applied
        {"content": "kept three", "active_form": ["not", "a", "str"]},  # bad active_form coerced
    ]
    assert parse_todo_items(payload) == (
        TodoItem(content="kept", status="in_progress", active_form="Keeping"),
        TodoItem(content="kept too"),
        TodoItem(content="kept three", active_form=""),
    )


def test_parse_never_raises_on_arbitrary_garbage() -> None:
    for raw in (
        object(),
        [object()],
        [None, [], {}],
        [{"content": None, "status": None}],
        [{"content": "x", "status": []}],  # unhashable status reaches the membership check
        [{"content": "x", "status": {}}],
        b"bytes",
        [set()],
    ):
        parse_todo_items(raw)  # must not raise

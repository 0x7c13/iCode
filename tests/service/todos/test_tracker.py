# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for TodoTracker — immutable-tuple state and the sync-reader contract."""

from __future__ import annotations

import inspect

import pytest

from chrys.foundation.models.todos import TodoItem
from chrys.service.todos.tracker import TodoTracker


@pytest.mark.asyncio
async def test_replace_returns_applied_snapshot() -> None:
    tracker = TodoTracker()
    items = (TodoItem(content="a"), TodoItem(content="b", status="in_progress", active_form="Doing b"))
    applied = await tracker.replace(items)
    assert applied == items
    assert tracker.snapshot() == items


@pytest.mark.asyncio
async def test_serialize_restore_round_trip() -> None:
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="x", status="completed"), TodoItem(content="y")))
    payload = tracker.serialize()

    other = TodoTracker()
    restored = await other.restore(payload)
    assert restored == tracker.snapshot()
    assert other.serialize() == payload


@pytest.mark.asyncio
async def test_restore_is_tolerant_of_garbage() -> None:
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="stale"),))

    assert await tracker.restore("not a list") == ()
    assert tracker.snapshot() == ()

    restored = await tracker.restore(["junk", {"content": "kept"}, {"content": "", "status": "bogus"}])
    assert restored == (TodoItem(content="kept"),)


@pytest.mark.asyncio
async def test_clear_drops_all_items() -> None:
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="a"),))
    await tracker.clear()
    assert tracker.snapshot() == ()
    assert tracker.serialize() == []


def test_empty_tracker_reads() -> None:
    tracker = TodoTracker()
    assert tracker.snapshot() == ()
    assert tracker.serialize() == []


def test_readers_are_sync() -> None:
    """snapshot()/serialize() must stay plain sync calls.

    SystemReminderMiddleware providers are sync callables invoked without
    ``await`` and the engine save path is sync — turning a reader into a
    coroutine breaks both silently (an un-awaited coroutine is truthy).
    """
    assert not inspect.iscoroutinefunction(TodoTracker.snapshot)
    assert not inspect.iscoroutinefunction(TodoTracker.serialize)


@pytest.mark.asyncio
async def test_readers_are_lock_free() -> None:
    """Readers must not take the writer lock — they are called from sync code
    that cannot await, potentially while a writer holds the lock."""
    tracker = TodoTracker()
    items = (TodoItem(content="a"),)
    await tracker.replace(items)
    async with tracker._lock:
        assert tracker.snapshot() == items
        assert tracker.serialize() == [{"content": "a", "status": "pending", "active_form": ""}]

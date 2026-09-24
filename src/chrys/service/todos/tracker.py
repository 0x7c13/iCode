# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""TodoTracker — engine-owned, session-scoped todo-list state."""

from __future__ import annotations

import asyncio

from chrys.foundation.models.todos import TodoItem, parse_todo_items, serialize_todo_items


class TodoTracker:
    """Holds the current session todo list as one immutable tuple.

    Concurrency contract:

    - **Writers** (:meth:`replace`, :meth:`clear`, :meth:`restore`) are async
      and serialize on an ``asyncio.Lock``.
    - **Readers** (:meth:`snapshot`, :meth:`serialize`) are sync and lock-free:
      they read the current tuple reference, which is only ever swapped, never
      mutated, so the read is atomic. This is required, not style —
      ``SystemReminderMiddleware`` providers are sync callables invoked without
      ``await``, and the engine save path is sync; neither can take the lock.
    """

    def __init__(self) -> None:
        self._items: tuple[TodoItem, ...] = ()
        self._lock = asyncio.Lock()

    async def replace(self, items: tuple[TodoItem, ...]) -> tuple[TodoItem, ...]:
        """Replace the whole list, returning the applied snapshot.

        Callers publish ``TodoListUpdated`` from the returned snapshot AFTER
        this call returns (never while holding the lock — ``EventBus.publish``
        awaits subscriber callbacks inline).
        """
        async with self._lock:
            self._items = tuple(items)
            return self._items

    async def clear(self) -> None:
        """Drop all items."""
        async with self._lock:
            self._items = ()

    async def restore(self, raw: object) -> tuple[TodoItem, ...]:
        """Hydrate from a persisted ``chrys_todos`` payload; tolerant, never raises."""
        async with self._lock:
            self._items = parse_todo_items(raw)
            return self._items

    def snapshot(self) -> tuple[TodoItem, ...]:
        """Current list (sync, lock-free)."""
        return self._items

    def serialize(self) -> list[dict[str, str]]:
        """Current list as the plain-dict ``chrys_todos`` payload (sync, lock-free)."""
        return serialize_todo_items(self._items)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Build-local bounded LRU and cancellation-safe, task-free single-flight locks."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    status: int
    content_type: str
    byte_count: int
    text: str
    redirect_to: str | None = None


class FetchCache:
    """Cache converted documents only, at most 64 entries/16 MiB/15 minutes."""

    def __init__(self) -> None:
        self.entries: OrderedDict[str, tuple[float, FetchedPage, int]] = OrderedDict()
        self.bytes = 0
        self.locks: dict[str, tuple[asyncio.Lock, int]] = {}

    def get(self, key: str) -> FetchedPage | None:
        value = self.entries.get(key)
        if value is None:
            return None
        if time.monotonic() - value[0] >= 900:
            self.entries.pop(key)
            self.bytes -= value[2]
            return None
        self.entries.move_to_end(key)
        return value[1]

    def put(self, key: str, page: FetchedPage) -> None:
        size = len(page.text.encode("utf-8")) + len(key.encode("utf-8")) + len(page.final_url.encode("utf-8")) + 512
        if size > 16 * 1024 * 1024 or page.redirect_to is not None:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[2]
        self.entries[key] = (time.monotonic(), page, size)
        self.bytes += size
        while len(self.entries) > 64 or self.bytes > 16 * 1024 * 1024:
            self.bytes -= self.entries.popitem(last=False)[1][2]

    @asynccontextmanager
    async def flight(self, key: str):
        lock, users = self.locks.get(key, (asyncio.Lock(), 0))
        self.locks[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            remaining = self.locks[key][1] - 1
            if remaining:
                self.locks[key] = (lock, remaining)
            else:
                self.locks.pop(key)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual message-pump close patch."""

from __future__ import annotations

import asyncio

from textual.message_pump import MessagePump

from chrys.foundation.patches.textual_message_pump import apply_runtime_patch


async def test_runtime_patch_waits_for_already_closing_message_pump() -> None:
    """A later ``wait=True`` close must await a task after ``wait=False`` started closing."""
    apply_runtime_patch()
    pump = MessagePump()
    release = asyncio.Event()

    async def runner() -> None:
        await release.wait()

    pump._task = asyncio.create_task(runner())

    await pump._close_messages(wait=False)
    assert pump._closing is True

    waiter = asyncio.create_task(pump._close_messages(wait=True))
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    await asyncio.wait_for(waiter, timeout=1)
    await asyncio.wait_for(pump._task, timeout=1)


async def test_runtime_patch_waits_for_closed_but_unfinished_message_pump() -> None:
    """``_closed`` can be true before ``_message_loop_exit`` has completed."""
    apply_runtime_patch()
    pump = MessagePump()
    release = asyncio.Event()

    async def runner() -> None:
        await release.wait()

    pump._task = asyncio.create_task(runner())
    pump._closed = True

    waiter = asyncio.create_task(pump._close_messages(wait=True))
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    await asyncio.wait_for(waiter, timeout=1)
    await asyncio.wait_for(pump._task, timeout=1)

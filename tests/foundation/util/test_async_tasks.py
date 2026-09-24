# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared asyncio task-boundary helpers."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.util.async_tasks import await_task_quiescence


@pytest.mark.asyncio
async def test_await_task_quiescence_tolerates_captured_task_cancellation() -> None:
    task = asyncio.create_task(asyncio.Event().wait())
    task.cancel()

    await await_task_quiescence(task)


@pytest.mark.asyncio
async def test_await_task_quiescence_propagates_waiter_cancellation() -> None:
    captured = asyncio.create_task(asyncio.Event().wait())
    waiter = asyncio.create_task(await_task_quiescence(captured))
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert captured.done() is False
    captured.cancel()
    await asyncio.gather(captured, return_exceptions=True)

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared asyncio task-boundary helpers."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def await_task_quiescence(task: asyncio.Task[None] | None) -> None:
    """Await one captured task while distinguishing caller cancellation.

    The captured task's cancellation or failure still establishes quiescence;
    cancellation of the waiting task propagates to its caller.
    """
    if task is None or task is asyncio.current_task():
        return
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        waiter = asyncio.current_task()
        if waiter is not None and waiter.cancelling() > 0:
            raise
        logger.debug("Captured task was cancelled before quiescence", exc_info=True)
    except Exception:
        logger.debug("Captured task failed before quiescence", exc_info=True)

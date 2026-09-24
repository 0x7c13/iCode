# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Buddy lifecycle callbacks for application entrypoints."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from chrys.app.features.buddy.actions import record_turn

if TYPE_CHECKING:
    from chrys.app.features.buddy.model import Buddy

logger = logging.getLogger(__name__)

_FAILED = "Failed to credit the buddy with a turn"


def on_successful_turn() -> None:
    """Credit the buddy with a finished turn. A buddy must never be why a turn fails, or why it waits.

    Every frontend finishes its turns on the event loop, and crediting waits for the save file's
    lock, which another instance may hold for seconds. So the write goes to the loop's default
    executor, which the loop's runner joins at exit, and the turn goes on at once.
    """
    try:
        credit = asyncio.get_running_loop().run_in_executor(None, record_turn)
    except Exception:
        logger.debug(_FAILED, exc_info=True)
        return
    credit.add_done_callback(_log_a_failure)


def _log_a_failure(credit: asyncio.Future[Buddy | None]) -> None:
    if credit.cancelled():
        return
    failure = credit.exception()
    if failure is not None:
        logger.debug(_FAILED, exc_info=failure)

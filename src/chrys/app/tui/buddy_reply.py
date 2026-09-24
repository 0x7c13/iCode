# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Petting the buddy from any TUI surface: a thinking toast now, its answer when the model has one."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from typing import TYPE_CHECKING

from chrys.app.features.buddy.replies import pet_reply, reply_gate
from chrys.app.tui.buddy_messages import THINKING_LINES

if TYPE_CHECKING:
    from collections.abc import Callable

    from chrys.app.features.buddy.model import Buddy
    from chrys.foundation.i18n import MessageRef

logger = logging.getLogger(__name__)

_THINKING_TOAST_SECONDS = 5
_ANSWER_TOAST_SECONDS = 10


class PetReplyFlow:
    """One surface's share of the process-wide reply gate, and the task answering through it."""

    def __init__(self, toast: Callable[[MessageRef | str, float], None], *, on_answered: Callable[[], None]) -> None:
        self._toast = toast
        self._on_answered = on_answered
        self._task: asyncio.Task[None] | None = None
        self._holds_gate = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The answer being waited for, if any."""
        return self._task

    @property
    def answering(self) -> bool:
        """Whether an answer is on its way, from here or anywhere else."""
        return self._task is not None or reply_gate.locked()

    def start(self, buddy: Buddy) -> bool:
        """Ask *buddy* for an answer. False when one is already on its way, from here or anywhere else."""
        if not reply_gate.acquire(blocking=False):
            return False
        self._holds_gate = True
        try:
            self._toast(random.choice(THINKING_LINES).bind(name=buddy.display_name), _THINKING_TOAST_SECONDS)  # noqa: S311
            self._task = asyncio.create_task(self._answer(buddy))
        except BaseException:
            self._release_gate()
            raise
        return True

    async def _answer(self, buddy: Buddy) -> None:
        try:
            self._toast(await pet_reply(buddy), _ANSWER_TOAST_SECONDS)
            self._on_answered()
        except Exception:
            # Nobody awaits this task, and a surface that went away mid-answer is not worth a traceback.
            logger.debug("Buddy answer could not be shown", exc_info=True)
        finally:
            self._task = None
            self._release_gate()

    async def shutdown(self) -> None:
        """Stop waiting for the answer."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # A task cancelled before its first step never reaches its own cleanup. A new answer that was
        # started while the old one was being awaited holds the gate in its own right.
        if self._task is None:
            self._release_gate()

    def _release_gate(self) -> None:
        if self._holds_gate:
            self._holds_gate = False
            reply_gate.release()

# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: wait for already-closing Textual message pumps.

Problem
-------
Textual 8.2.5 treats ``MessagePump._close_messages()`` as a no-op once
``_closing`` is set.  A first close with ``wait=False`` can therefore make a
later cleanup close with ``wait=True`` return without waiting for the pump task.
That leaves internal widgets such as scrollbars vulnerable to asyncio's
``Task was destroyed but it is pending!`` warning during shutdown.

Solution
--------
Keep the first-close behavior, but let repeated close calls wait for the
existing task when the caller requested ``wait=True``.

Some Chrys TUI modules import Textual before ``bootstrap_runtime()`` applies
site-package patches, so this module also exposes an in-process monkey patch
for already-imported ``MessagePump`` classes.
"""

from __future__ import annotations

import asyncio
import contextlib
from asyncio import CancelledError
from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_OLD = """\
    async def _close_messages(self, wait: bool = True) -> None:
        \"\"\"Close message queue, and optionally wait for queue to finish processing.\"\"\"
        if self._closed or self._closing:
            return
        self._closing = True
        if self._timers:
            await Timer._stop_all(self._timers)
            self._timers.clear()
        Reactive._reset_object(self)
        self._message_queue.put_nowait(None)
        if wait and self._task is not None and asyncio.current_task() != self._task:
            try:
                running_widget = active_message_pump.get()
            except LookupError:
                running_widget = None

            if running_widget is None or running_widget is not self:
                try:
                    await self._task
                except CancelledError:
                    pass"""

_NEW = """\
    async def _close_messages(self, wait: bool = True) -> None:
        \"\"\"Close message queue, and optionally wait for queue to finish processing.\"\"\"

        async def wait_for_close_task() -> None:
            if wait and self._task is not None and asyncio.current_task() != self._task:
                try:
                    running_widget = active_message_pump.get()
                except LookupError:
                    running_widget = None

                if running_widget is None or running_widget is not self:
                    try:
                        await self._task
                    except CancelledError:
                        pass

        if self._closed or self._closing:
            await wait_for_close_task()
            return
        self._closing = True
        if self._timers:
            await Timer._stop_all(self._timers)
            self._timers.clear()
        Reactive._reset_object(self)
        self._message_queue.put_nowait(None)
        await wait_for_close_task()"""

_RUNTIME_PATCH_MARKER = "_chrys_waits_for_closing_message_pump"


def apply_runtime_patch() -> None:
    """Patch ``MessagePump._close_messages`` in the current process."""
    try:
        from textual._context import active_message_pump
        from textual.message_pump import MessagePump
        from textual.reactive import Reactive
        from textual.timer import Timer
    except ImportError:
        return

    if getattr(MessagePump._close_messages, _RUNTIME_PATCH_MARKER, False):
        return

    async def _close_messages(self: Any, wait: bool = True) -> None:
        async def wait_for_close_task() -> None:
            if wait and self._task is not None and asyncio.current_task() != self._task:
                try:
                    running_widget = active_message_pump.get()
                except LookupError:
                    running_widget = None

                if running_widget is None or running_widget is not self:
                    with contextlib.suppress(CancelledError):
                        await self._task

        if self._closed or self._closing:
            await wait_for_close_task()
            return
        self._closing = True
        if self._timers:
            await Timer._stop_all(self._timers)
            self._timers.clear()
        Reactive._reset_object(self)
        self._message_queue.put_nowait(None)
        await wait_for_close_task()

    setattr(_close_messages, _RUNTIME_PATCH_MARKER, True)
    MessagePump._close_messages = _close_messages


register(
    FilePatch(
        package="textual",
        module_file="message_pump.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Wait for already-closing Textual message pump tasks",
    ),
)

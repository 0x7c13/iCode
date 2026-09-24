# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Widget removal that cancelling its caller cannot cut short.

Textual's removal awaitable gathers the removed widgets' message loops, so cancelling the wait
cancels them. A widget cancelled while it waits for its own children to exit never detaches: it
stays in the DOM, blank and without a message loop, and the App's message loop, which awaits the
same removal, ends as well. Code that a cancellable task can run (a worker, a flow task, or an
EventBus handler, which runs inside the publishing engine task) removes through these helpers.

A cancelled caller waits for the removal to finish before its cancellation goes on, so whatever runs
after it, and any lock it holds, sees the widgets already gone. Cancelling it again ends that wait at
once; the removal still finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from textual.widget import Widget


async def finish_shielded(operation: Awaitable[object]) -> None:
    """Await *operation* (a removal such as ``Tabs.clear()``, or a coroutine built on one) to its end."""
    task = asyncio.ensure_future(operation)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # The cancellation outranks whatever the operation raises.
        with contextlib.suppress(Exception):
            await asyncio.shield(task)
        raise


async def remove_shielded(widget: Widget) -> None:
    """Remove *widget* to the end, even when the caller is cancelled."""
    await finish_shielded(widget.remove())


async def remove_children_shielded(container: Widget) -> None:
    """Remove *container*'s children to the end, even when the caller is cancelled."""
    await finish_shielded(container.remove_children())

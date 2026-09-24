# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Streaming markdown updater — batches fragments for efficient rendering."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown


class MarkdownStream:
    """An object to manage streaming markdown.

    This will accumulate markdown fragments if they can't be rendered fast enough.

    This object is typically created by the VirtualizedMarkdown.get_stream class method.
    """

    def __init__(self, markdown_widget: VirtualizedMarkdown) -> None:
        """
        Args:
            markdown_widget: VirtualizedMarkdown widget to update.
        """
        self.markdown_widget = markdown_widget
        self._task: asyncio.Task | None = None
        self._new_markup = asyncio.Event()
        self._pending: list[str] = []
        self._stopped = False

    def start(self) -> None:
        """Start the updater running in the background."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the stream and await its finish."""
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None
            self._stopped = True

    async def write(self, markdown_fragment: str) -> None:
        """Append or enqueue a markdown fragment.

        Args:
            markdown_fragment: A string to append at the end of the document.
        """
        if self._stopped:
            raise RuntimeError("Can't write to the stream after it has stopped.")
        if not markdown_fragment:
            return
        self._pending.append(markdown_fragment)
        self._new_markup.set()
        await asyncio.sleep(0)

    async def _run(self) -> None:
        """Run a task to append markdown fragments when available."""
        try:
            while await self._new_markup.wait():
                new_markdown = "".join(self._pending)
                self._pending.clear()
                self._new_markup.clear()
                await asyncio.shield(self.markdown_widget.append(new_markdown))
        except asyncio.CancelledError:
            pass

        new_markdown = "".join(self._pending)
        if new_markdown:
            await self.markdown_widget.append(new_markdown)

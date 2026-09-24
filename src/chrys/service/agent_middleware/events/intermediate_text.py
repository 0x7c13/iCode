# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Intermediate assistant text buffering for tool-event ordering."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from chrys.service.trajectory.preparation import preparation_lock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from chrys.kernel import Content, Message
    from chrys.service.trajectory.preparation import PreparationTrace


def intermediate_text_contents(messages: Sequence[Message]) -> tuple[Content, ...]:
    """Return the text of one response that the intermediate-text callback publishes.

    A response that calls a local tool without provider-hosted content
    narrates through the client's callback: at once when blocking, through
    ``IntermediateTextBuffer`` when streaming. The hosted presentation
    bridge publishes the text of every other response, so this rule decides
    which of the two owns each text occurrence.
    """
    text_contents: list[Content] = []
    has_function_calls = False
    for message in messages:
        for content in message.contents:
            if content.provider_hosted:
                return ()
            if content.type == "text":
                if content.text:
                    text_contents.append(content)
            elif content.type == "function_call" and not content.informational_only:
                has_function_calls = True
    return tuple(text_contents) if has_function_calls else ()


class IntermediateTextBuffer:
    """Buffer for intermediate text detected by result_hook.

    The result_hook (sync, cannot await) stores text here, and ``release``
    publishes it ahead of whatever follows the response that wrote it: the
    response's first tool start, the next response, or the end of the pass
    for a call that starts no tool (an unknown tool, or arguments rejected
    before the tool pipeline).

    Also tracks a ``batch_id`` counter that increments once per LLM
    response.  The middleware reads this to tag session messages with
    the correct batch grouping.
    """

    def __init__(self) -> None:
        self._pending: list[str] = []
        self._publication: asyncio.Future[None] | None = None
        self._release_lock = asyncio.Lock()
        self.batch_id: int = 0

    def new_batch(self) -> None:
        """Signal a new LLM response (batch boundary)."""
        self.batch_id += 1

    def store(self, text: str) -> None:
        """Store intermediate text (called from sync result_hook)."""
        self.new_batch()
        self._pending.append(text)

    def drain(self) -> list[str]:
        """Take all pending text without publishing it."""
        items = self._pending.copy()
        self._pending.clear()
        return items

    async def release(
        self, publish: Callable[[str], Awaitable[None]], preamble: PreparationTrace | None = None
    ) -> None:
        """Publish all pending text in order; a no-op when nothing is pending.

        Parallel tool calls of one response each release before publishing
        their start. The lock holds every sibling until the text is fully
        published, so no start of the batch precedes it.

        A text leaves the buffer only when its publication starts, and a
        started publication runs to completion even if the releasing task
        is cancelled: publishing it again would repeat it to the subscribers
        that already received it. The next release, at the latest ``finish``
        at the end of the pass, waits for it before the text still pending.
        """
        async with preparation_lock(self._release_lock, preamble):
            while self._publication is not None or self._pending:
                if self._publication is None:
                    self._publication = asyncio.ensure_future(publish(self._pending.pop(0)))
                publication = self._publication
                try:
                    await asyncio.shield(publication)
                finally:
                    if publication.done():
                        self._publication = None

    async def finish(self, publish: Callable[[str], Awaitable[None]]) -> None:
        """Release what is left as a pass ends; nothing this buffer started outlives the call.

        The pass end waits for a publication that a cancelled release left
        running, unless the pass itself is being cancelled: its owner is
        closing it, and a subscriber that never returns must not hold the
        close. Then, and when this release is cancelled while it waits, the
        publication is cancelled and the text still pending is dropped.
        """
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            await self._abandon()
            return
        try:
            await self.release(publish)
        except asyncio.CancelledError:
            await self._abandon()
            raise

    async def _abandon(self) -> None:
        self._pending.clear()
        publication, self._publication = self._publication, None
        if publication is not None:
            publication.cancel()
            # Settled, not only asked to stop: nothing reaches a subscriber after the pass.
            await asyncio.gather(publication, return_exceptions=True)

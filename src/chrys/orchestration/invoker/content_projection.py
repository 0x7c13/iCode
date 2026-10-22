# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Publish an invocation's model output as content segments while it arrives.

One :class:`ContentProjector` per caller pass is the only publisher of answer
and reasoning text. Streamed updates go out as they arrive (:meth:`on_update`).
When a response lands (:meth:`response_ready`, awaited by the kernel loop
before it runs the response's tools), a response that streamed nothing
readable — a blocking request, or the blocking fallback of a stalled stream —
goes out whole, and every open segment closes. A tool call or result, or a
change of channel, closes the open segment; the next text opens a new one.
The provider-hosted work a landed response ran waits for the text before it
(the ``shown`` callback), so it shows in the response's order.

Answer text that was shown is never sent again: :attr:`exposed` tells a retry
owner that replaying the pass would repeat it, and
:meth:`take_unlanded_answer` hands over the answer a failed turn keeps: what
the response being streamed showed before it failed, and the text the last
landed response showed, which history may already hold. Every answer text
content that goes out is stamped with its segment id
(:data:`ANSWER_SEGMENT_KEY`), so the history that keeps it can later name
what to withdraw. Reasoning may stream again: a retry shows a fresh run of
thought, never a repeated answer.
A retry gives up the response being streamed (:meth:`on_retry_boundary`): one
``InvocationContentAbandoned`` names its segments before the open one closes
as failed, and none of its text is kept.

A caller whose views show each answer segment whole once it completes
(``shows_whole_segments``: chat sub-agents) counts streamed answer text as
shown only then (:attr:`response_exposed` for the wire call being streamed):
a failure before that may retry it unseen, and a failed pass keeps only the
segments that completed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from chrys.foundation.events.types import InvocationContentAbandoned, InvocationContentDelta, InvocationContentFinished
from chrys.foundation.models.history_markers import ANSWER_SEGMENT_KEY
from chrys.kernel import TOOL_CALL_CONTENT_TYPES, TOOL_RESULT_CONTENT_TYPES
from chrys.service.llm.reasoning_display import readable_reasoning

if TYPE_CHECKING:
    from chrys.foundation.models.invocations import ContentChannel, ContentStatus
    from chrys.kernel import ChatResponse, Content

    from .origin import BoundEmitter

_BOUNDARY_TYPES = TOOL_CALL_CONTENT_TYPES | TOOL_RESULT_CONTENT_TYPES


class _HasContents(Protocol):
    @property
    def contents(self) -> Sequence[Content]: ...


@dataclass(slots=True)
class _Segment:
    segment_id: str
    channel: ContentChannel
    # Whether it holds answer text that is not blank, and its text contents:
    # what a whole-segment caller shows once it completes.
    readable: bool = False
    texts: list[Content] = field(default_factory=list)


class ContentProjector:
    """The content segments of one invocation pass, published through its bound emitter."""

    def __init__(self, emitter: BoundEmitter, *, agent_name: str = "", shows_whole_segments: bool = False) -> None:
        self._emitter = emitter
        self._agent_name = agent_name
        self._whole_segments = shows_whole_segments
        # While a landed response whose answer is kept whole already goes out.
        self._landing = False
        self._response_exposed = False
        self._open: _Segment | None = None
        # Contents of the response being streamed, by identity: a content
        # object emitted twice is one fragment, as response assembly reads it.
        # The memo holds them strongly so a recycled id never matches.
        self._seen: dict[int, Content] = {}
        self._streamed = False
        # The segments of the response being streamed: a retry gives them all up.
        self._response_segments: list[str] = []
        self._unlanded_answer: list[Content] = []
        self.exposed = False
        """Whether any answer text of this pass was shown."""

    @property
    def response_exposed(self) -> bool:
        """Whether answer text of the response being streamed was shown."""
        return self._response_exposed

    async def on_update(self, update: _HasContents) -> None:
        """Publish one streamed update's readable text."""
        for content in update.contents:
            if content.type == "usage" or id(content) in self._seen:
                continue
            self._seen[id(content)] = content
            if await self._project(content):
                self._streamed = True

    async def response_ready(
        self, response: ChatResponse, *, shown: Callable[[Content | None], Awaitable[None]] | None = None
    ) -> None:
        """A response landed: publish it whole unless it streamed, then close its segments.

        *shown* hears each content of a response that did not stream once it went out, then ``None``
        once the whole response did.
        """
        streamed = self._streamed
        self._reset_response()
        answers = [message for message in response.messages if message.role == "assistant"]
        # History stores a response only once the loop records it or the run's
        # providers run: a pass cancelled before then keeps the text it
        # showed, which history may already hold. A response that streamed
        # showed its landed text (the fragments' stamps carry over); one that
        # did not is kept as it goes out, or, by a whole-segment caller, as
        # its segments complete: one cancelled midway showed only those.
        self._landing = streamed or not self._whole_segments
        if streamed:
            self._unlanded_answer = [
                content
                for message in answers
                for content in message.contents
                if content.type == "text" and ANSWER_SEGMENT_KEY in content.additional_properties
            ]
        try:
            if not streamed:
                for message in answers:
                    for content in message.contents:
                        await self._project(content)
                        if shown is not None:
                            await shown(content)
            await self._close("completed")
            if shown is not None:
                await shown(None)
        finally:
            # The next response is another wire call.
            self._landing = self._response_exposed = False
            self._response_segments = []

    async def on_retry_boundary(self) -> None:
        """The response being streamed is abandoned for a fresh attempt."""
        abandoned = tuple(self._response_segments)
        # Dropped first: a cancel while the events go out keeps none of the text they give up.
        self._reset_response()
        if abandoned:
            await self._emitter.publish(
                InvocationContentAbandoned(
                    origin=self._emitter.origin,
                    session_id=self._emitter.origin.session_id,
                    agent_name=self._agent_name,
                    segment_ids=abandoned,
                )
            )
        await self._close("failed")

    async def publish_text(self, channel: ContentChannel, text: str) -> str | None:
        """Publish plain *text* on *channel*, for a backend that streams text rather than contents; name its segment."""
        return await self._append(channel, text) if text else None

    async def end_segment(self, status: ContentStatus = "completed") -> None:
        """Close the open segment with *status*; the next text opens a new one."""
        await self._close(status)

    async def finish(self, status: ContentStatus) -> None:
        """End the pass: close the open segment with *status*."""
        await self._close(status)

    def take_unlanded_answer(self) -> tuple[Content, ...]:
        """Hand over, once, the answer contents this pass showed that history may not hold."""
        answer, self._unlanded_answer = tuple(self._unlanded_answer), []
        return answer

    def _reset_response(self) -> None:
        self._seen.clear()
        self._streamed = False
        self._response_segments = []
        self._unlanded_answer = []
        self._response_exposed = False

    def _expose(self) -> None:
        self.exposed = self._response_exposed = True

    async def _project(self, content: Content) -> bool:
        """Publish one content's readable text; True when some went out."""
        if content.type == "text":
            if content.text:
                await self._append("answer", content.text, content)
                return True
        elif content.type == "text_reasoning":
            if (shown := readable_reasoning(content)) is not None:
                await self._append(*shown)
                return True
        elif content.type in _BOUNDARY_TYPES:
            await self._close("completed")
        return False

    async def _append(self, channel: ContentChannel, text: str, content: Content | None = None) -> str:
        segment = self._open
        if segment is None or segment.channel != channel:
            await self._close("completed")
            segment = self._open = _Segment(uuid4().hex, channel)
            self._response_segments.append(segment.segment_id)
        readable = channel == "answer" and not text.isspace()
        if content is not None:
            # A fresh dict: the content's own may be shared with another copy.
            content.additional_properties = {**content.additional_properties, ANSWER_SEGMENT_KEY: segment.segment_id}
            if not self._whole_segments:
                # Kept before the publish: a subscriber may show it before a cancel cuts the publish short.
                self._unlanded_answer.append(content)
        if self._whole_segments:
            segment.readable = segment.readable or readable
            if content is not None:
                segment.texts.append(content)
        elif readable:
            # Set before the publish: an inline subscriber may already show
            # the text when the publish fails or is cancelled.
            self._expose()
        await self._emitter.publish(
            InvocationContentDelta(
                origin=self._emitter.origin,
                session_id=self._emitter.origin.session_id,
                agent_name=self._agent_name,
                segment_id=segment.segment_id,
                channel=channel,
                text=text,
            )
        )
        return segment.segment_id

    async def _close(self, status: ContentStatus) -> None:
        segment, self._open = self._open, None
        if segment is None:
            return
        if self._whole_segments and status == "completed" and segment.readable:
            # The segment shows as this publish goes out.
            self._expose()
            if not self._landing:
                self._unlanded_answer.extend(segment.texts)
        await self._emitter.publish(
            InvocationContentFinished(
                origin=self._emitter.origin,
                session_id=self._emitter.origin.session_id,
                agent_name=self._agent_name,
                segment_id=segment.segment_id,
                channel=segment.channel,
                status=status,
            )
        )

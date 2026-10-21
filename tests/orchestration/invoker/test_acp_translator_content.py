# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ACP translator streams the remote agent's message and thought text as content segments."""

from __future__ import annotations

from functools import partial

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationContentDelta, InvocationContentFinished
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.invoker.acp_protocol import AcpUpdateTranslator
from tests.orchestration.sub_agents._acp_fakes import session_notification
from tests.support.event_capture import collect_events

_ORIGIN = InvocationOrigin("sub_agent", "parent", "inv", None)

type _ContentEvent = InvocationContentDelta | InvocationContentFinished


def _translator(bus: EventBus | None) -> AcpUpdateTranslator:
    return AcpUpdateTranslator(
        event_bus=bus,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=1,
        origin=_ORIGIN,
    )


async def _content_events(bus: EventBus) -> list[_ContentEvent]:
    events: list[_ContentEvent] = []
    await bus.subscribe(InvocationContentDelta, partial(collect_events, events))
    await bus.subscribe(InvocationContentFinished, partial(collect_events, events))
    return events


def _message(text: str, message_id: str = "m1") -> AgentMessageChunk:
    return AgentMessageChunk(
        sessionUpdate="agent_message_chunk", messageId=message_id, content=TextContentBlock(type="text", text=text)
    )


def _thought(text: str) -> AgentThoughtChunk:
    return AgentThoughtChunk(sessionUpdate="agent_thought_chunk", content=TextContentBlock(type="text", text=text))


def _shape(events: list[_ContentEvent]) -> list[tuple[str, str, str]]:
    """Each event as (kind, channel, text or status)."""
    return [
        ("delta", event.channel, event.text)
        if isinstance(event, InvocationContentDelta)
        else ("finished", event.channel, event.status)
        for event in events
    ]


async def test_message_and_thought_chunks_stream_as_answer_and_reasoning_under_the_translator_origin() -> None:
    bus = EventBus()
    events = await _content_events(bus)
    translator = _translator(bus)

    for seq, update in enumerate([_thought("Weighing it."), _message("Hello "), _message("world")], start=1):
        await translator.put(seq, session_notification(update))
    await translator.finish_content("completed")

    # A change of channel ends the reasoning segment; the answer's chunks share one segment.
    assert _shape(events) == [
        ("delta", "reasoning", "Weighing it."),
        ("finished", "reasoning", "completed"),
        ("delta", "answer", "Hello "),
        ("delta", "answer", "world"),
        ("finished", "answer", "completed"),
    ]
    assert [event.segment_id for event in events] == [events[0].segment_id] * 2 + [events[2].segment_id] * 3
    assert events[0].segment_id != events[2].segment_id
    assert {(event.origin, event.session_id, event.agent_name) for event in events} == {(_ORIGIN, "parent", "External")}
    # Thoughts are shown, never returned.
    assert translator.result_text() == "Hello world"


async def test_a_tool_call_start_or_a_new_message_ends_the_open_segment() -> None:
    bus = EventBus()
    events = await _content_events(bus)
    translator = _translator(bus)

    await translator.put(1, session_notification(_message("Reading. ")))
    # Progress of a running call merges tool state and leaves the segment open.
    await translator.put(
        2,
        session_notification(
            ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="early", title="Read", status="in_progress")
        ),
    )
    await translator.put(3, session_notification(_message("Still reading. ")))
    await translator.put(
        4,
        session_notification(
            ToolCallStart(sessionUpdate="tool_call", toolCallId="call", title="Read", kind="read", status="in_progress")
        ),
    )
    await translator.put(5, session_notification(_message("Found it. ")))
    await translator.put(6, session_notification(_message("Done.", message_id="m2")))

    assert _shape(events) == [
        ("delta", "answer", "Reading. "),
        ("delta", "answer", "Still reading. "),
        ("finished", "answer", "completed"),
        ("delta", "answer", "Found it. "),
        ("finished", "answer", "completed"),
        ("delta", "answer", "Done."),
    ]
    segments = [event.segment_id for event in events]
    assert segments[0] == segments[1] == segments[2]
    assert len({segments[0], segments[3], segments[5]}) == 3


async def test_flush_interrupted_finishes_the_open_segment_as_interrupted() -> None:
    bus = EventBus()
    events = await _content_events(bus)
    translator = _translator(bus)

    await translator.put(1, session_notification(_message("partial")))
    await translator.flush_interrupted()
    # The segment already ended: a later finish has nothing left to close.
    await translator.finish_content("completed")

    assert _shape(events) == [("delta", "answer", "partial"), ("finished", "answer", "interrupted")]
    assert events[1].segment_id == events[0].segment_id

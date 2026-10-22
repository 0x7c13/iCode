# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ACP translator streams the remote agent's message and thought text as content segments."""

from __future__ import annotations

from functools import partial

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationContentDelta, InvocationContentFinished, InvocationToolCallStart
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.invoker import acp_protocol
from chrys.orchestration.invoker.acp_protocol import AcpUpdateTranslator
from chrys.service.acp_client.errors import AcpTransportError
from chrys.service.session.sub_agent_logs import ACP_ANSWER_WITHDRAWN_UPDATE
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

    await translator.put(
        0,
        session_notification(
            ToolCallStart(
                sessionUpdate="tool_call", toolCallId="early", title="Read", kind="read", status="in_progress"
            )
        ),
    )
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


@pytest.mark.parametrize("opens_card", ["start", "update"])
async def test_the_final_answer_is_every_answer_segment_shown_after_the_last_tool_card_opened(opens_card: str) -> None:
    """What a retry in a new session asks for again; a later update of an open card opens none, ending nothing."""
    bus = EventBus()
    events = await _content_events(bus)
    translator = _translator(bus)
    if opens_card == "start":
        opened: ToolCallStart | ToolCallProgress = ToolCallStart(
            sessionUpdate="tool_call", toolCallId="call", title="Read", kind="read", status="in_progress"
        )
    else:
        # An update for a call not seen yet opens its card too.
        opened = ToolCallProgress(
            sessionUpdate="tool_call_update", toolCallId="call", title="Read", status="in_progress"
        )

    await translator.put(1, session_notification(_message("Reading. ")))
    await translator.put(2, session_notification(opened))
    # The card ends the segment, though the message goes on.
    await translator.put(3, session_notification(_message("Found it. ")))
    await translator.put(
        4,
        session_notification(
            ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="call", title="Read", rawInput={"path": "a"})
        ),
    )
    await translator.put(5, session_notification(_message("Done.", message_id="m3")))

    answers = list(dict.fromkeys(event.segment_id for event in events if isinstance(event, InvocationContentDelta)))
    assert len(answers) == 3
    assert translator.withdraw_final_answer() == (answers[1], answers[2])
    # The audit records the withdrawal once, so a replay drops that answer too.
    assert translator.withdraw_final_answer() == ()
    withdrawals = [
        item
        for item in translator.translated_updates
        if item["update"] == {"sessionUpdate": ACP_ANSWER_WITHDRAWN_UPDATE}
    ]
    assert withdrawals == [{"seq": None, "update": {"sessionUpdate": ACP_ANSWER_WITHDRAWN_UPDATE}, "attempt": 1}]


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


async def test_a_card_the_interrupt_flush_opens_leaves_the_cut_off_answer_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call cut off before its card opened gets one as the pass ends; that card ends no segment completed."""
    monkeypatch.setattr(acp_protocol, "_MAX_ASSEMBLED_BYTES", 1_000)
    bus = EventBus()
    events = await _content_events(bus)
    starts: list[InvocationToolCallStart] = []
    await bus.subscribe(InvocationToolCallStart, partial(collect_events, starts))
    translator = _translator(bus)
    over_budget = ToolCallProgress(
        sessionUpdate="tool_call_update", toolCallId="call", title="Read", rawInput={"path": "x" * 2_000}
    )

    await translator.put(1, session_notification(_message("Half an")))
    # The update records the call, then fails over budget before its card opens.
    with pytest.raises(AcpTransportError):
        await translator.put(2, session_notification(over_budget))
    await translator.flush_interrupted()

    assert [start.call_id for start in starts] == ["a1:call"]
    assert _shape(events) == [("delta", "answer", "Half an"), ("finished", "answer", "interrupted")]

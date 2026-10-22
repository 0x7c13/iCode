# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The content projector: one publisher of answer and reasoning segments per pass."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Event,
    InvocationContentAbandoned,
    InvocationContentDelta,
    InvocationContentFinished,
)
from chrys.foundation.models.history_markers import ANSWER_SEGMENT_KEY
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message
from chrys.orchestration.invoker.content_projection import ContentProjector
from chrys.orchestration.invoker.origin import BoundEmitter
from tests.support.waiting import wait_for

ORIGIN = InvocationOrigin("turn", "s1", "main", None)


async def _projector(
    agent_name: str = "", *, shows_whole_segments: bool = False
) -> tuple[ContentProjector, list[Event]]:
    bus = EventBus()
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    await bus.subscribe(InvocationContentDelta, record)
    await bus.subscribe(InvocationContentFinished, record)
    await bus.subscribe(InvocationContentAbandoned, record)
    projector = ContentProjector(
        BoundEmitter(bus, ORIGIN), agent_name=agent_name, shows_whole_segments=shows_whole_segments
    )
    return projector, events


def _shape(events: list[Event]) -> list[tuple[str, ...]]:
    """Each event as (kind, channel, text-or-status), with segments numbered in order of appearance."""
    numbers: dict[str, int] = {}
    shaped: list[tuple[str, ...]] = []
    for event in events:
        if isinstance(event, InvocationContentDelta):
            number = numbers.setdefault(event.segment_id, len(numbers))
            shaped.append(("delta", str(number), event.channel, event.text))
        elif isinstance(event, InvocationContentFinished):
            shaped.append(("finished", str(numbers[event.segment_id]), event.channel, event.status))
        elif isinstance(event, InvocationContentAbandoned):
            shaped.append(("abandoned", *(str(numbers[segment_id]) for segment_id in event.segment_ids)))
    return shaped


def _update(*contents: Content) -> ChatResponseUpdate:
    return ChatResponseUpdate(role="assistant", contents=list(contents))


def _call(call_id: str = "c1") -> Content:
    return Content.from_function_call(call_id=call_id, name="read_file", arguments="{}")


async def test_streamed_text_splits_at_tool_calls_and_channel_changes_and_never_repeats() -> None:
    projector, events = await _projector()
    reasoning, first, second = (
        Content.from_text_reasoning(text="think"),
        Content.from_text("Let me "),
        Content.from_text("look."),
    )

    await projector.on_update(_update(reasoning, first))
    await projector.on_update(_update(second, _call()))
    await projector.response_ready(
        ChatResponse(messages=[Message("assistant", [reasoning, Content.from_text("Let me look."), _call()])])
    )
    await projector.on_update(_update(Content.from_text("Done.")))
    await projector.response_ready(ChatResponse(messages=[Message("assistant", [Content.from_text("Done.")])]))

    assert _shape(events) == [
        ("delta", "0", "reasoning", "think"),
        ("finished", "0", "reasoning", "completed"),
        ("delta", "1", "answer", "Let me "),
        ("delta", "1", "answer", "look."),
        ("finished", "1", "answer", "completed"),
        ("delta", "2", "answer", "Done."),
        ("finished", "2", "answer", "completed"),
    ]
    assert all(event.origin == ORIGIN and event.session_id == "s1" for event in events)


async def test_a_response_that_streamed_nothing_readable_goes_out_whole_when_it_lands() -> None:
    projector, events = await _projector(agent_name="Explore")

    # A stalled stream's only fragment was a call delta; its blocking fallback lands whole.
    await projector.on_update(_update(_call()))
    await projector.response_ready(
        ChatResponse(
            messages=[
                Message(
                    "assistant",
                    [
                        Content.from_text_reasoning(text="plan", additional_properties={"reasoning_summary": True}),
                        Content.from_text("Reading."),
                        _call(),
                        Content.from_text("Then more."),
                    ],
                ),
            ]
        )
    )

    assert _shape(events) == [
        ("delta", "0", "reasoning_summary", "plan"),
        ("finished", "0", "reasoning_summary", "completed"),
        ("delta", "1", "answer", "Reading."),
        ("finished", "1", "answer", "completed"),
        ("delta", "2", "answer", "Then more."),
        ("finished", "2", "answer", "completed"),
    ]
    assert {event.agent_name for event in events if isinstance(event, InvocationContentDelta)} == {"Explore"}


async def test_a_content_object_emitted_twice_goes_out_once() -> None:
    projector, events = await _projector()
    text = Content.from_text("once")

    await projector.on_update(_update(text))
    await projector.on_update(_update(text))

    assert _shape(events) == [("delta", "0", "answer", "once")]


async def test_opaque_reasoning_exposes_nothing() -> None:
    projector, events = await _projector()

    await projector.on_update(_update(Content.from_text_reasoning(text=None, protected_data="signature")))
    await projector.on_update(_update(Content.from_text_reasoning(id="rs", text="", protected_data="encrypted")))

    assert events == []
    assert projector.exposed is False


async def test_a_failed_pass_closes_its_segment_and_keeps_the_unlanded_answer() -> None:
    projector, events = await _projector()

    await projector.on_update(_update(Content.from_text_reasoning(text="hmm"), Content.from_text(" ")))
    # Reasoning and blank text may show again on a retry; answer text may not.
    assert projector.exposed is False
    await projector.on_update(_update(Content.from_text("Partial "), Content.from_text("answer")))
    assert projector.exposed is True
    await projector.finish("failed")

    assert _shape(events)[-1] == ("finished", "1", "answer", "failed")
    assert "".join(content.text or "" for content in projector.take_unlanded_answer()) == " Partial answer"
    # Handed over once: a second repair of the same pass keeps nothing more.
    assert projector.take_unlanded_answer() == ()


async def test_a_landed_response_keeps_the_text_it_showed_and_a_retry_boundary_keeps_none() -> None:
    """The loop may not have recorded a response with calls yet: its shown narration stays kept until history holds it."""
    projector, events = await _projector()
    updates = [_update(Content.from_text("Let me "), Content.from_text("look.")), _update(_call())]

    for update in updates:
        await projector.on_update(update)
    # The landed response the kernel assembles from the streamed updates.
    landed = ChatResponse.from_updates(updates)
    await projector.response_ready(landed)
    (narration,) = projector.take_unlanded_answer()
    assert narration is landed.messages[0].contents[0]
    assert narration.text == "Let me look."
    (delta, *_) = [event for event in events if isinstance(event, InvocationContentDelta)]
    assert narration.additional_properties[ANSWER_SEGMENT_KEY] == delta.segment_id

    await projector.on_update(_update(Content.from_text("abandoned")))
    await projector.on_retry_boundary()
    assert projector.take_unlanded_answer() == ()
    assert _shape(events)[-1] == ("finished", "1", "answer", "failed")


async def test_a_retry_boundary_gives_up_every_segment_of_the_streamed_response_only() -> None:
    """The response being streamed is given up whole, before its open segment closes as failed."""
    projector, events = await _projector()

    await projector.on_update(_update(Content.from_text("Let me look."), _call()))
    await projector.response_ready(
        ChatResponse(messages=[Message("assistant", [Content.from_text("Let me look."), _call()])])
    )
    # A blocking response goes out whole as it lands.
    await projector.response_ready(
        ChatResponse(messages=[Message("assistant", [Content.from_text("Again."), _call()])])
    )
    await projector.on_update(_update(Content.from_text_reasoning(text="hmm"), Content.from_text("Partial")))
    await projector.on_retry_boundary()

    assert _shape(events) == [
        ("delta", "0", "answer", "Let me look."),
        ("finished", "0", "answer", "completed"),
        ("delta", "1", "answer", "Again."),
        ("finished", "1", "answer", "completed"),
        ("delta", "2", "reasoning", "hmm"),
        ("finished", "2", "reasoning", "completed"),
        ("delta", "3", "answer", "Partial"),
        # The landed responses' segments are not given up; the completed reasoning is.
        ("abandoned", "2", "3"),
        ("finished", "3", "answer", "failed"),
    ]

    # A response that showed nothing gives up nothing.
    await projector.on_update(_update(Content.from_text("")))
    await projector.on_retry_boundary()
    assert len(events) == 9


@pytest.mark.parametrize("slow", [InvocationContentAbandoned, InvocationContentFinished])
async def test_a_retry_boundary_cancelled_while_it_gives_up_the_response_keeps_none_of_it(
    slow: type[Event],
) -> None:
    bus = EventBus()
    projector = ContentProjector(BoundEmitter(bus, ORIGIN))
    publishing, release = asyncio.Event(), asyncio.Event()

    async def slow_subscriber(_event: Event) -> None:
        publishing.set()
        await release.wait()

    await bus.subscribe(slow, slow_subscriber)
    await projector.on_update(_update(Content.from_text("Partial")))
    boundary = asyncio.create_task(projector.on_retry_boundary())
    await wait_for(lambda: publishing.is_set() or boundary.done(), description="the boundary publishes")
    assert not boundary.done()
    boundary.cancel()
    with pytest.raises(asyncio.CancelledError):
        await boundary

    assert projector.take_unlanded_answer() == ()


@pytest.mark.parametrize(
    "beside",
    [
        pytest.param((), id="alone"),
        pytest.param((Content.from_search_tool_call("ws1", tool_name="web_search"),), id="hosted-call"),
        pytest.param(
            (Content.from_function_call("ws2", "web_search", arguments={}, informational_only=True),),
            id="informational-call",
        ),
    ],
)
async def test_a_final_answer_stays_unlanded_until_the_pass_hands_it_over(beside: tuple[Content, ...]) -> None:
    """History stores a final answer only once the run's providers run: a pass cancelled before then keeps it.

    Calls the provider ran, or only reported, leave the answer final.
    """
    projector, _events = await _projector()
    updates = [_update(*beside), _update(Content.from_text("final"))]

    for update in updates:
        await projector.on_update(update)
    landed = ChatResponse.from_updates(updates)
    await projector.response_ready(landed)

    assert projector.take_unlanded_answer() == (landed.messages[0].contents[-1],)
    assert projector.take_unlanded_answer() == ()


async def test_a_response_cancelled_while_its_segment_closes_stays_unlanded() -> None:
    """The loop records a response after its segments close: a pass cancelled there keeps it."""
    bus = EventBus()
    projector = ContentProjector(BoundEmitter(bus, ORIGIN))
    closing, release = asyncio.Event(), asyncio.Event()

    async def slow_subscriber(_event: Event) -> None:
        closing.set()
        await release.wait()

    await bus.subscribe(InvocationContentFinished, slow_subscriber)
    updates = [_update(Content.from_text("lan"), Content.from_text("ded")), _update(_call())]
    for update in updates[:1]:
        await projector.on_update(update)
    response = ChatResponse.from_updates(updates)
    landed = response.messages[0].contents[0]
    ready = asyncio.create_task(projector.response_ready(response))
    await closing.wait()
    assert not ready.done()
    ready.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ready

    # The landed content, not the streamed fragments: history may hold it already.
    assert projector.take_unlanded_answer() == (landed,)


async def test_answer_text_a_cancel_cuts_off_mid_publish_stays_unlanded() -> None:
    """A subscriber may already show the text when the publish is cancelled."""
    bus = EventBus()
    projector = ContentProjector(BoundEmitter(bus, ORIGIN))
    showing, release = asyncio.Event(), asyncio.Event()

    async def slow_subscriber(_event: Event) -> None:
        showing.set()
        await release.wait()

    await bus.subscribe(InvocationContentDelta, slow_subscriber)
    shown = Content.from_text("Hello ")
    update = asyncio.create_task(projector.on_update(_update(shown)))
    await showing.wait()
    assert not update.done()
    update.cancel()
    with pytest.raises(asyncio.CancelledError):
        await update

    assert projector.take_unlanded_answer() == (shown,)
    assert projector.exposed is True


async def test_whole_segment_answer_text_is_shown_only_once_its_segment_completes() -> None:
    """Sub-agent views show each answer segment whole: a failed pass keeps only the segments that completed."""
    projector, events = await _projector(shows_whole_segments=True)
    first, second = Content.from_text("Let me "), Content.from_text("look.")

    await projector.on_update(_update(first, second))
    assert (projector.exposed, projector.response_exposed) == (False, False)
    # A hosted card starts: the segment before it completes and shows.
    await projector.end_segment()
    assert (projector.exposed, projector.response_exposed) == (True, True)
    await projector.on_update(_update(Content.from_text("Cut o")))
    await projector.finish("failed")

    assert _shape(events) == [
        ("delta", "0", "answer", "Let me "),
        ("delta", "0", "answer", "look."),
        ("finished", "0", "answer", "completed"),
        ("delta", "1", "answer", "Cut o"),
        ("finished", "1", "answer", "failed"),
    ]
    assert projector.take_unlanded_answer() == (first, second)


async def test_a_whole_segment_cut_off_by_a_retry_was_never_shown() -> None:
    projector, events = await _projector(shows_whole_segments=True)

    await projector.on_update(_update(Content.from_text_reasoning(text="hmm"), Content.from_text("Partial")))
    # The reasoning segment completed; reasoning may show again.
    assert (projector.exposed, projector.response_exposed) == (False, False)
    await projector.on_retry_boundary()
    assert projector.exposed is False
    assert projector.take_unlanded_answer() == ()
    assert _shape(events)[-1] == ("finished", "1", "answer", "failed")

    update = _update(Content.from_text("again"))
    await projector.on_update(update)
    landed = ChatResponse.from_updates([update])
    (final,) = landed.messages[0].contents
    await projector.response_ready(landed)

    assert projector.exposed is True
    # The next response is another wire call.
    assert projector.response_exposed is False
    # The landed answer once, not again for the segment it closed.
    assert projector.take_unlanded_answer() == (final,)


@pytest.mark.parametrize("whole", [False, True], ids=["streaming_view", "whole_segment_view"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["landed", "cancelled_at_hosted_card"])
async def test_a_blocking_response_keeps_the_segments_it_showed(cancelled: bool, whole: bool) -> None:
    """An interrupt while a hosted card opens between two segments showed only the first, stamped with its segment."""
    projector, events = await _projector(shows_whole_segments=whole)
    held, release = asyncio.Event(), asyncio.Event()
    narration, trailing = Content.from_text("Searching."), Content.from_text("Found it.")
    hosted = Content.from_search_tool_call("ws1", tool_name="web_search")

    async def shown(content: Content | None) -> None:
        if content is hosted and cancelled:
            held.set()
            await release.wait()

    response = ChatResponse(messages=[Message("assistant", [narration, hosted, trailing])])
    ready = asyncio.create_task(projector.response_ready(response, shown=shown))
    if cancelled:
        await held.wait()
        assert not ready.done()
        ready.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ready
    else:
        await ready

    # The landed contents, which history may hold already.
    assert projector.take_unlanded_answer() == ((narration,) if cancelled else (narration, trailing))
    segments = [event.segment_id for event in events if isinstance(event, InvocationContentDelta)]
    assert [content.additional_properties.get(ANSWER_SEGMENT_KEY) for content in (narration, trailing)] == (
        [segments[0], None] if cancelled else segments
    )

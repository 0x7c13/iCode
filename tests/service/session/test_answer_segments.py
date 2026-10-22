# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``SessionHistoryManager`` keeps a failed pass's shown answer by segment and names what a bare Continue drops.

The segment stamp is what joins history to frontends: what a Continue drops
from history is exactly the text it names, and frontends withdraw only that.
"""

from __future__ import annotations

from typing import Any

from chrys.foundation.models.history_markers import ANSWER_SEGMENT_KEY, HistoryMarkerKind
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.kernel import Content, Message
from chrys.service.session.history import SessionHistoryManager


def _shown(text: str, segment: str) -> Content:
    return Content.from_text(text, additional_properties={ANSWER_SEGMENT_KEY: segment})


def _hosted() -> list[Content]:
    return [
        Content.from_hosted_tool_call(
            "search-1",
            tool_name="web_search",
            arguments={"query": "Chrys"},
            status="running",
            hosted_family="search",
            hosted_provider="openai",
            provider_phase="start",
        ),
        Content.from_hosted_tool_result(
            "search-1",
            tool_name="web_search",
            result="found",
            status="completed",
            hosted_family="search",
            hosted_provider="openai",
            provider_phase="terminal",
        ),
    ]


def _history(*messages: Message) -> tuple[SessionHistoryManager, list[Message]]:
    state: dict[str, Any] = {"messages": [Message("user", ["go"]), *messages]}
    history = SessionHistoryManager()
    history.bind(state)
    return history, state["messages"]


def _segments(message: Message) -> list[tuple[str | None, object]]:
    return [(content.text, content.additional_properties.get(ANSWER_SEGMENT_KEY)) for content in message.contents]


def test_a_kept_partial_answer_is_one_stamped_text_per_segment_it_was_shown_in() -> None:
    history, messages = _history()

    history.keep_partial_answer(
        [_shown("Let me ", "intro"), _shown("look.", "intro"), _shown("  ", "blank"), _shown("The result", "answer")]
    )

    assert _segments(messages[-1]) == [("Let me look.", "intro"), ("The result", "answer")]
    assert messages[-1].additional_properties[ANALYTICS_ITEM_ID_KEY]


def test_continue_drops_the_trailing_answer_messages_and_names_their_segments_once() -> None:
    """Text before a provider-hosted call whose response never landed is kept as answer, and goes with it."""
    history, messages = _history(
        Message("assistant", [_shown("Searching.", "intro"), _shown("Found ", "answer")]),
        Message("assistant", [_shown("Chrys.", "answer")]),
    )

    assert history.remove_trailing_answer() == ("intro", "answer")
    assert [message.role for message in messages] == ["user"]


def test_continue_drops_only_the_answer_after_an_exchange_message_s_last_tool_content() -> None:
    narration, (call, result), answer = _shown("Searching.", "intro"), _hosted(), _shown("Found Chrys.", "answer")
    landed = Message("assistant", [narration, call, result, answer])
    item_id = landed.additional_properties[ANALYTICS_ITEM_ID_KEY] = "item-1"
    history, messages = _history(landed)

    assert history.remove_trailing_answer() == ("answer",)

    trimmed = messages[-1]
    assert all(kept is original for kept, original in zip(trimmed.contents, [narration, call, result], strict=True))
    assert trimmed.additional_properties[ANALYTICS_ITEM_ID_KEY] == item_id
    # The landed wrapper may be held elsewhere (a retry snapshot): it is replaced, never trimmed in place.
    assert trimmed is not landed
    assert list(landed.contents) == [narration, call, result, answer]


def test_continue_keeps_an_exchange_whose_tail_is_reasoning_only() -> None:
    call, result = _hosted()
    landed = Message("assistant", [call, result, Content.from_text_reasoning(text="hmm")])
    history, messages = _history(landed)

    assert history.remove_trailing_answer() == ()
    assert messages[-1] is landed


def test_continue_stops_at_any_marker_whatever_its_value() -> None:
    marker = Message("assistant", [_shown("Interrupted", "status")])
    marker.additional_properties[HistoryMarkerKind.KEY] = ""
    history, messages = _history(Message("assistant", [_shown("Done.", "earlier")]), marker)

    assert history.remove_trailing_answer() == ()
    assert messages[-1] is marker


def test_continue_drops_an_answer_kept_without_its_segment_and_names_none() -> None:
    """History saved before answers were stamped: the text still goes, and no frontend text is withdrawn."""
    history, messages = _history(Message("assistant", ["The result is"]))

    assert history.remove_trailing_answer() == ()
    assert [message.role for message in messages] == ["user"]

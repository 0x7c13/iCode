# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for OpenAI Responses reasoning parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.kernel import ChatResponse, Content
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.history import encode_content
from chrys.service.llm.openai_responses.stream import StreamState


def _stream() -> StreamState:
    return StreamState({}, model="gpt-test", variant=OPENAI_RESPONSES)


def _reasoning_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _reasoning_item(
    *,
    item_id: str = "rs_1",
    content: list[SimpleNamespace] | None = None,
    summary: list[SimpleNamespace] | None = None,
    encrypted_content: str | None = "encrypted-payload",
) -> SimpleNamespace:
    return SimpleNamespace(
        type="reasoning",
        id=item_id,
        content=content or [],
        summary=summary or [],
        encrypted_content=encrypted_content,
    )


def _response_with(item: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="gpt-test",
        metadata={},
        output=[item],
        usage=None,
        conversation=None,
        status="completed",
        incomplete_details=None,
    )


def _event(event_type: str, **attributes: object) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **attributes)


def test_non_streaming_reasoning_payload_survives_persistence_round_trip() -> None:
    item = _reasoning_item(
        content=[_reasoning_part("Private reasoning")],
        summary=[_reasoning_part("Visible summary")],
    )

    parsed = decode_response(_response_with(item), {}, variant=OPENAI_RESPONSES)
    restored = ChatResponse.from_dict(parsed.to_dict())
    reasoning_contents = restored.messages[0].contents

    assert [content.text for content in reasoning_contents] == ["Private reasoning", "Visible summary"]
    assert reasoning_contents[0].additional_properties["reasoning_text"] is True
    assert reasoning_contents[1].additional_properties == {}
    assert [content.protected_data for content in reasoning_contents] == ["encrypted-payload", None]


def test_streaming_reasoning_snapshot_stamps_every_visible_sibling() -> None:
    item = _reasoning_item(
        content=[_reasoning_part("First private part"), _reasoning_part("Second private part")],
        summary=[_reasoning_part("First summary"), _reasoning_part("Second summary")],
    )

    update = _stream().update_for(_event("response.output_item.added", item=item, output_index=0))

    assert [content.text for content in update.contents] == [
        "First private part",
        "Second private part",
        "First summary",
        "Second summary",
    ]
    assert [content.protected_data for content in update.contents] == ["encrypted-payload"] * 4


def test_streaming_reasoning_snapshot_emits_content_and_summary_siblings() -> None:
    item = _reasoning_item(
        content=[_reasoning_part("Private reasoning")],
        summary=[_reasoning_part("Visible summary")],
    )

    update = _stream().update_for(_event("response.output_item.added", item=item, output_index=0))
    response = ChatResponse.from_updates([update])

    assert [(content.text, content.additional_properties) for content in response.messages[0].contents] == [
        ("Private reasoning", {"reasoning_text": True}),
        ("Visible summary", {}),
    ]


@pytest.mark.parametrize("event_kind", ["delta", "done_fallback", "snapshot"])
def test_streamed_reasoning_text_paths_mark_private_text(event_kind: str) -> None:
    state = _stream()

    if event_kind == "delta":
        event = _event(
            "response.reasoning_text.delta",
            item_id="rs_delta",
            delta="Delta reasoning",
        )
    elif event_kind == "done_fallback":
        event = _event(
            "response.reasoning_text.done",
            item_id="rs_done",
            text="Fallback reasoning",
        )
    else:
        event = _event(
            "response.output_item.added",
            item=_reasoning_item(
                item_id="rs_snapshot",
                content=[_reasoning_part("Snapshot reasoning")],
                summary=[],
            ),
            output_index=0,
        )

    update = state.update_for(event)

    assert len(update.contents) == 1
    assert update.contents[0].additional_properties == {"reasoning_text": True}


def test_streamed_reasoning_terminal_done_emits_encrypted_payload_marker() -> None:
    item = _reasoning_item(item_id="rs_encrypted", encrypted_content="terminal-payload")

    update = _stream().update_for(_event("response.output_item.done", item=item))

    assert [(content.id, content.text, content.protected_data) for content in update.contents] == [
        ("rs_encrypted", "", "terminal-payload")
    ]


def test_streamed_summary_and_reasoning_text_remain_distinct_after_coalescing() -> None:
    state = _stream()
    events = [
        _event(
            "response.reasoning_summary_text.delta",
            item_id="rs_mixed",
            delta="Visible summary",
        ),
        _event(
            "response.reasoning_text.delta",
            item_id="rs_mixed",
            delta="Private reasoning",
        ),
        _event(
            "response.output_item.done",
            item=_reasoning_item(
                item_id="rs_mixed",
                encrypted_content="encrypted-mixed",
            ),
        ),
    ]

    response = ChatResponse.from_updates([state.update_for(event) for event in events])
    reasoning_contents = response.messages[0].contents

    assert [(content.text, content.additional_properties) for content in reasoning_contents] == [
        ("Visible summary", {}),
        ("Private reasoning", {"reasoning_text": True}),
    ]
    assert [content.protected_data for content in reasoning_contents] == [None, "encrypted-mixed"]


def test_legacy_reasoning_payload_properties_remain_serializable() -> None:
    legacy = Content.from_text_reasoning(
        id="rs_legacy",
        text="Visible summary",
        additional_properties={
            "encrypted_content": "legacy-payload",
            "reasoning_text": "Private reasoning",
        },
    )

    prepared = encode_content("assistant", legacy, provider="openai")

    assert prepared["encrypted_content"] == "legacy-payload"
    assert prepared["content"] == [{"type": "reasoning_text", "text": "Private reasoning"}]
    assert prepared["summary"] == [{"type": "summary_text", "text": "Visible summary"}]


def test_marker_true_reasoning_serializes_text_as_content_not_summary() -> None:
    marked = Content.from_text_reasoning(
        id="rs_marked",
        text="Private reasoning",
        additional_properties={
            "encrypted_content": "legacy-payload",
            "reasoning_text": True,
        },
    )

    prepared = encode_content("assistant", marked, provider="openai")

    assert prepared["encrypted_content"] == "legacy-payload"
    assert prepared["content"] == [{"type": "reasoning_text", "text": "Private reasoning"}]
    assert prepared["summary"] == []

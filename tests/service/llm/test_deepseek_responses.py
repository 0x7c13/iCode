# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""DeepSeek Responses dialect reasoning, statelessness, and wire-boundary tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest

from chrys.foundation.hosted_tools import PRESENTATION_TEXT_SEGMENT_ID_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.kernel.exceptions import ChatClientInvalidRequestException
from chrys.service.llm.deepseek import DeepSeekResponsesClient, DeepSeekResponsesReasoningReplayMode
from chrys.service.llm.openai_responses import RawOpenAIChatClient


class _FakeAsyncOpenAI:
    base_url = "https://api.deepseek.test"


def _client(mode: DeepSeekResponsesReasoningReplayMode = "plaintext-replay") -> DeepSeekResponsesClient:
    class _ModeClient(DeepSeekResponsesClient):
        REASONING_REPLAY_MODE: ClassVar[DeepSeekResponsesReasoningReplayMode] = mode

    return _ModeClient(model="deepseek-test", async_client=_FakeAsyncOpenAI())


def _base_client() -> RawOpenAIChatClient:
    return RawOpenAIChatClient(model="openai-test", async_client=_FakeAsyncOpenAI())


def _reasoning(
    *,
    text: str = "private",
    reasoning_id: str | None = "rs_1",
    payload: str | None = None,
    reasoning_text: bool = False,
    marker: object | None = None,
) -> Content:
    properties: dict[str, object] = {}
    if reasoning_text:
        properties["reasoning_text"] = True
    if marker is not None:
        properties["openai_reasoning_format"] = marker
    return Content.from_text_reasoning(
        id=reasoning_id,
        text=text,
        protected_data=payload,
        additional_properties=properties,
    )


def _call() -> Content:
    return Content.from_function_call(
        "call_1",
        "lookup",
        arguments="{}",
        additional_properties={"fc_id": "fc_1"},
    )


def _replay(mode: DeepSeekResponsesReasoningReplayMode, reasoning: Content) -> list[dict[str, object]]:
    return _client(mode)._prepare_messages_for_openai(
        [Message("assistant", [reasoning, Content.from_text("visible"), _call()])],
        request_uses_service_side_storage=False,
    )


def _item_types(items: list[dict[str, object]]) -> list[str]:
    return [str(item["type"]) for item in items]


@pytest.mark.parametrize("source_provider", ["openai", "anthropic"])
@pytest.mark.parametrize("family", ["search", "generic"])
def test_deepseek_cross_provider_hosted_history_degrades_to_assistant_context(
    source_provider: str,
    family: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if family == "search":
        call = Content.from_search_tool_call(
            "foreign-1",
            tool_name="web_search",
            arguments={"query": "Chrys"},
            status="running",
            hosted_provider=source_provider,
            provider_phase="start",
        )
        result = Content.from_search_tool_result(
            "foreign-1",
            tool_name="web_search",
            result="found",
            status="completed",
            hosted_provider=source_provider,
            provider_phase="terminal",
        )
    else:
        call = Content.from_hosted_tool_call(
            "foreign-1",
            tool_name="remote_task",
            arguments={"task": "inspect"},
            status="running",
            hosted_provider=source_provider,
            provider_phase="start",
        )
        result = Content.from_hosted_tool_result(
            "foreign-1",
            tool_name="remote_task",
            result="found",
            status="completed",
            hosted_provider=source_provider,
            provider_phase="terminal",
        )

    with caplog.at_level("DEBUG", logger="chrys.service.agent_middleware.events.hosted_tools"):
        replayed = _client()._prepare_messages_for_openai(
            [Message("assistant", [call]), Message("assistant", [result])],
            request_uses_service_side_storage=False,
        )

    assert _item_types(replayed) == ["message"]
    assert replayed[0]["role"] == "assistant"
    summary = replayed[0]["content"][0]["text"]
    assert "[Provider-hosted tool context]" in summary
    assert "Status: completed" in summary
    assert 'Result: "found"' in summary
    assert "Degrading provider-hosted history to assistant context" in caplog.text


@pytest.mark.parametrize("mode", ["plaintext-replay", "plaintext-valid-drop"])
@pytest.mark.parametrize("marker", ["reasoning_content", "reasoning"])
def test_known_foreign_reasoning_marker_validly_drops_only_the_occurrence(
    mode: DeepSeekResponsesReasoningReplayMode,
    marker: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    foreign = _reasoning(payload='{"foreign":true}', marker=marker)

    with caplog.at_level(logging.WARNING, logger="chrys.service.llm.openai_responses"):
        prepared = _replay(mode, foreign)

    assert _item_types(prepared) == ["message", "function_call"]
    assert prepared[-1]["id"] == "fc_1"
    assert "Degraded stateless reasoning replay" not in caplog.text


@pytest.mark.parametrize(
    ("mode", "expected_types", "has_encrypted", "keeps_fc_id"),
    [
        ("encrypted", ["reasoning", "message", "function_call"], True, True),
        ("plaintext-replay", ["reasoning", "message", "function_call"], False, True),
        ("plaintext-valid-drop", ["message", "function_call"], False, True),
    ],
)
def test_composite_reasoning_precedence_ladder(
    mode: DeepSeekResponsesReasoningReplayMode,
    expected_types: list[str],
    has_encrypted: bool,
    keeps_fc_id: bool,
) -> None:
    prepared = _replay(mode, _reasoning(payload="encrypted", reasoning_text=True))

    assert _item_types(prepared) == expected_types
    reasoning_items = [item for item in prepared if item["type"] == "reasoning"]
    if reasoning_items:
        assert ("encrypted_content" in reasoning_items[0]) is has_encrypted
        assert reasoning_items[0]["content"] == [{"type": "reasoning_text", "text": "private"}]
    assert ("id" in prepared[-1]) is keeps_fc_id


@pytest.mark.parametrize(
    ("mode", "expected_reasoning", "keeps_fc_id"),
    [
        ("encrypted", True, True),
        ("plaintext-replay", False, True),
        ("plaintext-valid-drop", False, True),
    ],
)
def test_encrypted_only_reasoning_precedence_ladder(
    mode: DeepSeekResponsesReasoningReplayMode,
    expected_reasoning: bool,
    keeps_fc_id: bool,
) -> None:
    prepared = _replay(mode, _reasoning(text="", payload="encrypted"))

    assert any(item["type"] == "reasoning" for item in prepared) is expected_reasoning
    assert ("id" in prepared[-1]) is keeps_fc_id


@pytest.mark.parametrize(
    ("mode", "expected_reasoning", "keeps_fc_id"),
    [
        ("encrypted", False, False),
        ("plaintext-replay", True, True),
        ("plaintext-valid-drop", False, True),
    ],
)
def test_plaintext_only_reasoning_precedence_ladder(
    mode: DeepSeekResponsesReasoningReplayMode,
    expected_reasoning: bool,
    keeps_fc_id: bool,
) -> None:
    prepared = _replay(mode, _reasoning(reasoning_text=True))

    assert any(item["type"] == "reasoning" for item in prepared) is expected_reasoning
    assert ("id" in prepared[-1]) is keeps_fc_id


@pytest.mark.parametrize(
    ("mode", "keeps_fc_id"),
    [("encrypted", False), ("plaintext-replay", True), ("plaintext-valid-drop", True)],
)
def test_summary_only_reasoning_precedence_ladder(
    mode: DeepSeekResponsesReasoningReplayMode,
    keeps_fc_id: bool,
) -> None:
    prepared = _replay(mode, _reasoning())

    assert all(item["type"] != "reasoning" for item in prepared)
    assert ("id" in prepared[-1]) is keeps_fc_id


@pytest.mark.parametrize("mode", ["encrypted", "plaintext-replay", "plaintext-valid-drop"])
def test_idless_foreign_protected_payload_degrades_in_every_mode(
    mode: DeepSeekResponsesReasoningReplayMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    anthropic_shaped = _reasoning(reasoning_id=None, payload="anthropic-signature")

    with caplog.at_level(logging.WARNING, logger="chrys.service.llm.openai_responses"):
        prepared = _replay(mode, anthropic_shaped)

    assert _item_types(prepared) == ["message", "function_call"]
    assert "id" not in prepared[-1]
    assert "Degraded stateless reasoning replay" in caplog.text


@pytest.mark.parametrize("mode", ["encrypted", "plaintext-replay", "plaintext-valid-drop"])
def test_unrecognized_reasoning_marker_degrades_in_every_dialect_mode(
    mode: DeepSeekResponsesReasoningReplayMode,
) -> None:
    prepared = _replay(mode, _reasoning(payload="payload", marker="future_reasoning_field"))

    assert _item_types(prepared) == ["message", "function_call"]
    assert "id" not in prepared[-1]


@pytest.mark.parametrize(
    ("mode", "keeps_fc_id"),
    [("encrypted", False), ("plaintext-replay", True), ("plaintext-valid-drop", True)],
)
def test_payload_free_reasoning_marker_precedence_ladder(
    mode: DeepSeekResponsesReasoningReplayMode,
    keeps_fc_id: bool,
) -> None:
    prepared = _replay(mode, _reasoning(text=""))

    assert _item_types(prepared) == ["message", "function_call"]
    assert ("id" in prepared[-1]) is keeps_fc_id


def test_base_client_keeps_synthetic_id_bearing_foreign_marker_behavior() -> None:
    reasoning = _reasoning(payload="encrypted", marker="reasoning_content")

    prepared = _base_client()._prepare_messages_for_openai(
        [Message("assistant", [reasoning, _call()])],
        request_uses_service_side_storage=False,
    )

    assert _item_types(prepared) == ["reasoning", "function_call"]
    assert prepared[0]["encrypted_content"] == "encrypted"


def test_legacy_encrypted_content_location_obeys_plaintext_composite_policy() -> None:
    reasoning = _reasoning(reasoning_text=True)
    reasoning.additional_properties["encrypted_content"] = "legacy-encrypted"

    replayed = _replay("plaintext-replay", reasoning)
    dropped = _replay("plaintext-valid-drop", reasoning)

    assert _item_types(replayed) == ["reasoning", "message", "function_call"]
    assert replayed[0]["content"] == [{"type": "reasoning_text", "text": "private"}]
    assert "encrypted_content" not in replayed[0]
    assert _item_types(dropped) == ["message", "function_call"]
    assert dropped[-1]["id"] == "fc_1"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"store": True},
        {"store": False},
        {"extra_body": {"store": True}},
        {"extra_body": {"store": False}},
        {"extra_body": {"store": None}},
        {"store": True, "extra_body": {"store": False}},
        {"store": False, "extra_body": {"store": True}},
    ],
)
async def test_raw_client_normalizes_every_store_spelling_to_stateless_wire(options: dict[str, object]) -> None:
    prepared = await _client()._prepare_options([Message("user", ["hi"])], options)

    if options:
        assert prepared["store"] is False
    else:
        assert "store" not in prepared
    assert "store" not in prepared.get("extra_body", {})


async def test_raw_client_strips_top_level_and_nested_conversation_handles() -> None:
    prepared = await _client()._prepare_options(
        [Message("user", ["hi"])],
        {
            "conversation_id": "conv",
            "previous_response_id": "resp",
            "conversation": {"id": "thread"},
            "extra_body": {
                "conversation_id": "nested-conv",
                "previous_response_id": "nested-resp",
                "conversation": {"id": "nested-thread"},
            },
        },
    )

    assert not {
        "conversation_id",
        "previous_response_id",
        "conversation",
    }.intersection(prepared)
    assert prepared["extra_body"] == {}


async def test_raw_client_suppresses_automatic_include_but_preserves_caller_include() -> None:
    client = _client()

    automatic = await client._prepare_options([Message("user", ["hi"])], {})
    explicit = await client._prepare_options([Message("user", ["hi"])], {"include": ["file_search_call.results"]})

    assert "include" not in automatic
    assert explicit["include"] == ["file_search_call.results"]


@pytest.mark.parametrize("key", ["continuation_token", "background"])
@pytest.mark.parametrize("nested", [False, True])
def test_stateful_execution_options_are_rejected_at_the_raw_boundary(key: str, nested: bool) -> None:
    value: object = {"response_id": "resp_1"} if key == "continuation_token" else True
    options = {"extra_body": {key: value}} if nested else {key: value}

    with pytest.raises(ChatClientInvalidRequestException, match=key):
        _client()._inner_get_response(messages=[Message("user", ["hi"])], options=options)


def _response(*, status: str, output: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="deepseek-test",
        metadata={},
        output=output or [],
        usage=None,
        conversation=SimpleNamespace(id="conv_1"),
        status=status,
        incomplete_details=None,
    )


def test_blocking_parse_never_learns_conversation_or_continuation_handles() -> None:
    parsed = _client()._parse_response_from_openai(_response(status="in_progress"), {"store": True})

    assert parsed.conversation_id is None
    assert parsed.continuation_token is None


@pytest.mark.parametrize("event_type", ["response.created", "response.in_progress"])
def test_streaming_parse_never_learns_conversation_or_continuation_handles(event_type: str) -> None:
    update = _client()._parse_chunk_from_openai(
        SimpleNamespace(type=event_type, response=_response(status="in_progress")),
        {"store": True},
        {},
    )

    assert update.conversation_id is None
    assert update.continuation_token is None


def test_deepseek_cached_usage_is_extracted_for_responses() -> None:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=None,
        output_tokens_details=None,
        prompt_cache_hit_tokens=7,
        model_extra={},
    )

    details = _client()._parse_usage_from_openai(usage)

    assert details["deepseek.prompt_cache_hit_tokens"] == 7


@pytest.mark.parametrize("mode", ["plaintext-replay", "plaintext-valid-drop"])
def test_blocking_parse_persist_replay_obeys_plaintext_mode(mode: DeepSeekResponsesReasoningReplayMode) -> None:
    output = [
        SimpleNamespace(
            type="reasoning",
            id="rs_1",
            content=[SimpleNamespace(text="private")],
            summary=[],
            encrypted_content=None,
        ),
        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="working", annotations=[])]),
        SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="lookup",
            arguments="{}",
            status="completed",
        ),
    ]
    parsed = _client()._parse_response_from_openai(_response(status="completed", output=output), {})
    restored = ChatResponse.from_dict(parsed.to_dict())

    replayed = _client(mode)._prepare_messages_for_openai(
        restored.messages,
        request_uses_service_side_storage=False,
    )

    expected = ["reasoning", "message", "function_call"] if mode == "plaintext-replay" else ["message", "function_call"]
    assert _item_types(replayed) == expected
    assert replayed[-1]["id"] == "fc_1"


@pytest.mark.parametrize("mode", ["encrypted", "plaintext-replay", "plaintext-valid-drop"])
def test_blocking_composite_parse_persist_replay_obeys_precedence(
    mode: DeepSeekResponsesReasoningReplayMode,
) -> None:
    output = [
        SimpleNamespace(
            type="reasoning",
            id="rs_1",
            content=[SimpleNamespace(text="private")],
            summary=[],
            encrypted_content="encrypted",
        )
    ]
    parsed = _client()._parse_response_from_openai(_response(status="completed", output=output), {})
    restored = ChatResponse.from_dict(parsed.to_dict())

    replayed = _client(mode)._prepare_messages_for_openai(
        restored.messages,
        request_uses_service_side_storage=False,
    )

    if mode == "plaintext-valid-drop":
        assert replayed == []
    else:
        assert _item_types(replayed) == ["reasoning"]
        assert ("encrypted_content" in replayed[0]) is (mode == "encrypted")
        assert replayed[0]["content"] == [{"type": "reasoning_text", "text": "private"}]


@pytest.mark.parametrize("mode", ["plaintext-replay", "plaintext-valid-drop"])
def test_streaming_snapshot_delta_done_persist_replay_obeys_plaintext_mode(
    mode: DeepSeekResponsesReasoningReplayMode,
) -> None:
    parser = _client()
    snapshot = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        content=[],
        summary=[],
        encrypted_content=None,
    )
    updates = [
        parser._parse_chunk_from_openai(
            SimpleNamespace(type="response.output_item.added", item=snapshot, output_index=0),
            {},
            {},
        ),
        parser._parse_chunk_from_openai(
            SimpleNamespace(type="response.reasoning_text.delta", item_id="rs_1", delta="private"),
            {},
            {},
        ),
        parser._parse_chunk_from_openai(
            SimpleNamespace(type="response.reasoning_text.done", item_id="rs_1", text="private"),
            {},
            {},
            seen_reasoning_delta_item_ids={"rs_1"},
        ),
    ]
    restored = ChatResponse.from_dict(ChatResponse.from_updates(updates).to_dict())

    replayed = _client(mode)._prepare_messages_for_openai(
        restored.messages,
        request_uses_service_side_storage=False,
    )

    if mode == "plaintext-replay":
        assert _item_types(replayed) == ["reasoning"]
        assert replayed[0]["content"] == [{"type": "reasoning_text", "text": "private"}]
    else:
        assert replayed == []


@pytest.mark.parametrize("mode", ["encrypted", "plaintext-replay", "plaintext-valid-drop"])
def test_streaming_snapshot_terminal_composite_obeys_precedence(mode: DeepSeekResponsesReasoningReplayMode) -> None:
    parser = _client()
    snapshot = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        content=[SimpleNamespace(text="private")],
        summary=[],
        encrypted_content="snapshot-encrypted",
    )
    terminal = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        content=None,
        summary=[],
        encrypted_content="terminal-encrypted",
    )
    updates = [
        parser._parse_chunk_from_openai(
            SimpleNamespace(type="response.output_item.added", item=snapshot, output_index=0),
            {},
            {},
        ),
        parser._parse_chunk_from_openai(
            SimpleNamespace(type="response.output_item.done", item=terminal, output_index=0),
            {},
            {},
        ),
    ]
    restored = ChatResponse.from_dict(ChatResponse.from_updates(updates).to_dict())

    replayed = _client(mode)._prepare_messages_for_openai(
        restored.messages,
        request_uses_service_side_storage=False,
    )

    if mode == "plaintext-valid-drop":
        assert replayed == []
    else:
        assert _item_types(replayed) == ["reasoning"]
        assert ("encrypted_content" in replayed[0]) is (mode == "encrypted")
        assert replayed[0]["content"] == [{"type": "reasoning_text", "text": "private"}]


def _web_search_item(*, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        type="web_search_call",
        id="ws_1",
        status=status,
        action=SimpleNamespace(type="search", query="Chrys"),
    )


def _deepseek_streamed_search(item: SimpleNamespace) -> ChatResponse:
    client = _client()
    calls: dict[int, Content] = {}
    results: dict[int, Content] = {}
    updates = []
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(type="web_search_call", id="ws_1", status="in_progress", action=None),
        ),
        SimpleNamespace(type="response.web_search_call.searching", output_index=0),
        SimpleNamespace(type="response.web_search_call.completed", output_index=0),
        SimpleNamespace(type="response.output_item.done", output_index=0, item=item),
    ]
    for event in events:
        updates.append(
            client._parse_chunk_from_openai(
                event,
                {},
                {},
                hosted_call_contents=calls,
                hosted_result_contents=results,
            )
        )
    return ChatResponse.from_updates(updates)


def test_deepseek_blocking_web_search_uses_dialect_provider_id() -> None:
    parsed = _client()._parse_response_from_openai(
        _response(status="completed", output=[_web_search_item()]),
        {},
    )

    call, result = parsed.messages[0].contents
    assert (call.type, result.type) == ("search_tool_call", "search_tool_result")
    assert call.hosted_provider == result.hosted_provider == "deepseek-openai"
    assert call.retry_safety == result.retry_safety == "read_only"


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_deepseek_streaming_added_searching_completed_done_and_failure(status: str) -> None:
    item = _web_search_item(status=status)
    blocking = _client()._parse_response_from_openai(_response(status="completed", output=[item]), {})
    streaming = _deepseek_streamed_search(item)

    assert [content.to_dict() for content in streaming.messages[0].contents] == [
        content.to_dict() for content in blocking.messages[0].contents
    ]
    assert streaming.messages[0].contents[1].provider_status == status
    assert streaming.messages[0].contents[1].additional_properties.get("is_error", False) is (status == "failed")


def test_deepseek_intermediate_text_search_and_final_text_preserve_order() -> None:
    client = _client()
    calls: dict[int, Content] = {}
    results: dict[int, Content] = {}
    item = _web_search_item()
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Checking sources.",
            item_id="msg_1",
            output_index=0,
            content_index=0,
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(type="web_search_call", id="ws_1", status="in_progress", action=None),
        ),
        SimpleNamespace(type="response.web_search_call.searching", output_index=1),
        SimpleNamespace(type="response.output_item.done", output_index=1, item=item),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Final answer.",
            item_id="msg_2",
            output_index=2,
            content_index=0,
        ),
    ]
    updates = [
        client._parse_chunk_from_openai(
            event,
            {},
            {},
            hosted_call_contents=calls,
            hosted_result_contents=results,
        )
        for event in events
    ]

    response = ChatResponse.from_updates(updates)

    assert [content.type for content in response.messages[0].contents] == [
        "text",
        "search_tool_call",
        "search_tool_result",
        "text",
    ]
    assert response.messages[0].contents[0].text == "Checking sources."
    assert response.messages[0].contents[0].additional_properties[PRESENTATION_TEXT_SEGMENT_ID_KEY] == (
        "item:msg_1:content:0"
    )
    assert response.messages[0].contents[-1].text == "Final answer."


def test_deepseek_forced_stateless_search_history_round_trip() -> None:
    parsed = _client()._parse_response_from_openai(
        _response(status="completed", output=[_web_search_item()]),
        {},
    )
    restored = ChatResponse.from_dict(parsed.to_dict())

    replayed = _client()._prepare_messages_for_openai(
        restored.messages,
        request_uses_service_side_storage=False,
    )

    assert replayed == [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Chrys"},
        }
    ]

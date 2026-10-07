# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for per-agent image capability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chrys.foundation.text.images import is_image_media_type as common_is_image_media_type
from chrys.kernel import EXCLUDED_KEY, ChatContext, Content, Message, estimate_message_tokens, set_excluded
from chrys.kernel import is_image_media_type as kernel_is_image_media_type
from chrys.kernel.compaction import _token_count
from chrys.service.context.compaction import MixedLanguageTokenizer
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.state.store import JsonFileStateStore
from chrys.service.tools.builtins.filesystem import view_image
from chrys.service.vision import filter_image_tools, image_stub_middleware_for_model


@pytest.mark.parametrize(
    ("media_type", "expected"),
    [
        ("image/png", True),
        ("image/jpeg; charset=binary", True),
        (" IMAGE/webp ", True),
        ("image/svg+xml", True),
        ("image", True),
        ("image/", True),
        ("text/plain", False),
        ("application/json", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_image_media_type_predicates_take_any_image_type(media_type: object, expected: bool) -> None:
    assert kernel_is_image_media_type(media_type) is expected
    assert common_is_image_media_type(media_type) is expected


def test_filter_image_tools_removes_view_image_for_text_only_models() -> None:
    assert filter_image_tools([view_image], vision_enabled=False) == []
    assert filter_image_tools([view_image], vision_enabled=True) == [view_image]


async def test_non_vision_stub_preserves_tool_result_contents_and_writes_through_metadata() -> None:
    image = Content.from_data(
        b"image-bytes",
        "image/png",
        additional_properties={"width": 12, "height": 8, "media_type": "image/png"},
    )
    original_result = Content.from_function_result(
        "call_1",
        result=[Content.from_text("caption"), image],
    )
    original_message = Message("tool", [original_result], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)

    async def _final() -> None:
        set_excluded(context.messages[0], excluded=True, reason="budget_tool_compaction")

    await middleware.process(context, _final)

    assert context.messages[0] is not original_message
    assert original_message.additional_properties[EXCLUDED_KEY] is True
    assert original_result.items == [Content.from_text("caption"), image]
    replacement_result = context.messages[0].contents[0]
    assert replacement_result is not original_result
    assert replacement_result.items is not None
    assert [item.type for item in replacement_result.items] == ["text", "text"]
    assert "12x8 image/png omitted" in (replacement_result.items[1].text or "")
    assert original_result.items[1] is image


async def test_non_vision_stub_preserves_direct_image_contents_and_writes_through_metadata() -> None:
    image = Content.from_data(
        b"image-bytes",
        "image/png",
        additional_properties={"width": 1, "height": 2, "media_type": "image/png"},
    )
    original_message = Message("user", ["describe", image], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)

    async def _final() -> None:
        set_excluded(context.messages[0], excluded=True, reason="budget_tool_compaction")

    await middleware.process(context, _final)

    assert context.messages[0] is not original_message
    assert original_message.additional_properties[EXCLUDED_KEY] is True
    assert original_message.contents[1] is image
    assert context.messages[0].contents[1].type == "text"
    assert "1x2 image/png omitted" in (context.messages[0].contents[1].text or "")


async def test_non_vision_stub_ignores_unknown_uri_without_media_type() -> None:
    uri = Content.from_uri("https://example.com/blob")
    original_message = Message("user", [uri], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)

    async def _final() -> None:
        return None

    await middleware.process(context, _final)

    assert context.messages[0] is original_message
    assert context.messages[0].contents[0] is uri


@pytest.mark.parametrize("nested", [False, True], ids=["user_image", "tool_result_image"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("restore", [False, True], ids=["live_history", "restored_history"])
@pytest.mark.parametrize("start_vision", [False, True], ids=["text_first", "vision_first"])
async def test_image_token_counts_follow_capability_switches(tmp_path, nested, stream, restore, start_vision) -> None:
    image = Content.from_data(b"image-bytes", "image/png", additional_properties={"width": 2048, "height": 2048})
    if nested:
        history = [
            Message("user", ["inspect"]),
            Message("assistant", [Content.from_function_call("image_call", "image_tool")]),
            Message("tool", [Content.from_function_result("image_call", result=[image])]),
        ]
    else:
        history = [Message("user", ["inspect", image])]
    image_index = len(history) - 1
    tokenizer = MixedLanguageTokenizer()
    store = JsonFileStateStore(tmp_path)
    observed: dict[bool, list[int]] = {False: [], True: []}
    # Exercise both directions repeatedly, starting with either representation.
    for vision_enabled in (start_vision, not start_vision, start_vision, not start_vision, start_vision):
        middleware = image_stub_middleware_for_model(vision_enabled=vision_enabled)
        client = MockChatClient(responses=[MockResponse(text="done")], middleware=[middleware])
        if stream:
            response = client.get_response(history, stream=True, tokenizer=tokenizer)
            async for _ in response:
                pass
            await response.get_final_response()
        else:
            await client.get_response(history, tokenizer=tokenizer)
        wire_message = client.call_history[0][0][image_index]
        estimate = _token_count(wire_message)
        assert estimate == estimate_message_tokens(wire_message, tokenizer)
        assert estimate is not None
        observed[vision_enabled].append(estimate)
        # History retains the original image even while the model sees a stub.
        assert estimate_message_tokens(history[image_index], tokenizer) > 2000
        if restore:
            await store.save_session("capability-switch", {"messages": history}, agent_profile="Code")
            restored = await store.load_session("capability-switch")
            assert restored is not None
            history = restored["messages"]
    assert max(observed[False]) < min(observed[True]) // 10

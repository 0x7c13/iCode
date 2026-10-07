# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Replayed call arguments and outputs keep non-ASCII text readable on every wire."""

from __future__ import annotations

from chrys.kernel import Content, Message
from chrys.service.llm.chat_completions.client import OPENAI
from chrys.service.llm.chat_completions.history import encode_message
from chrys.service.llm.openai_responses.history import arguments_text, mcp_output_text


def test_chat_completions_replays_mapping_arguments_as_written() -> None:
    call = Content.from_function_call(call_id="c1", name="search", arguments={"q": "北京天气"})

    (prepared,) = encode_message(Message(role="assistant", contents=[call]), variant=OPENAI)

    assert prepared["tool_calls"][0]["function"]["arguments"] == '{"q": "北京天气"}'


def test_responses_replays_arguments_and_mcp_output_as_written() -> None:
    assert arguments_text({"q": "北京天气"}) == '{"q": "北京天气"}'
    assert mcp_output_text({"结果": "晴"}) == '{"结果": "晴"}'
    assert mcp_output_text([{"结果": "晴"}]) == '{"结果": "晴"}'

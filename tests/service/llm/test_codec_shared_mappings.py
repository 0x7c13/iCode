# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Codecs read caller and SDK mappings without writing into them.

A tool's ``parameters()`` is the cached schema local argument validation
reads, request options are reused across requests, and a decoded SDK
response may be read again: every codec edit lands on a copy.
"""

from __future__ import annotations

import copy
from typing import Any

from openai.types.responses import Response

from chrys.kernel import FunctionTool, Message
from chrys.service.llm.anthropic_messages.request import build_request
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.request import encode_tools
from tests.support.wire_cases._kit import RESP_USAGE_2, resp_message, resp_response


def _lookup(city: str, unit: str = "c") -> str:
    return f"{city}:{unit}"


async def test_responses_tool_encoding_leaves_the_schema_local_validation_reads() -> None:
    # A supplied schema that does not close its properties accepts extra
    # arguments locally; closing it for the request must not close it here.
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = FunctionTool(name="lookup", func=_lookup, input_model=schema)
    before = copy.deepcopy(tool.parameters())

    first, second = encode_tools([tool]), encode_tools([tool])

    assert first == second
    assert first[0]["parameters"] == {**before, "additionalProperties": False}
    assert tool.parameters() == before
    result = await tool.invoke(arguments={"city": "Paris", "unit": "f"})
    assert [content.text for content in result] == ["Paris:f"]


def test_anthropic_end_user_id_never_lands_in_the_callers_metadata() -> None:
    metadata = {"trace": "t-1"}
    messages = [Message(role="user", contents=["hi"])]

    first = build_request(
        messages,
        {"user": "user-1", "metadata": metadata},
        {},
        model="claude-test",
        base_url="https://api.anthropic.com",
        default_headers={},
    ).request
    second = build_request(
        messages,
        {"user": "user-2", "metadata": metadata},
        {},
        model="claude-test",
        base_url="https://api.anthropic.com",
        default_headers={},
    ).request

    assert metadata == {"trace": "t-1"}
    assert first["metadata"] == {"trace": "t-1", "user_id": "user-1"}
    assert second["metadata"] == {"trace": "t-1", "user_id": "user-2"}


def test_responses_decode_keeps_log_probabilities_off_the_sdk_response() -> None:
    message: dict[str, Any] = resp_message("msg_1", "Sunny.")
    logprobs = [{"token": "Sunny", "bytes": [83], "logprob": -0.1, "top_logprobs": []}]
    message["content"][0]["logprobs"] = logprobs
    # Validation requires every usage field the SDK models, cache writes included.
    usage = {**RESP_USAGE_2, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}
    response = Response.model_validate(
        {**resp_response(response_id="resp_1", output=[message], usage=usage), "metadata": {"trace": "t-1"}}
    )

    decoded = decode_response(response, {}, variant=OPENAI_RESPONSES)

    assert response.metadata == {"trace": "t-1"}
    assert decoded.additional_properties is not response.metadata
    assert decoded.additional_properties["trace"] == "t-1"
    assert [entry.model_dump() for entry in decoded.additional_properties["logprobs"]] == logprobs

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A content-filter rejection keeps its details whatever codes and severities the service reports."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from chrys.kernel import Message
from chrys.service.llm.chat_completions import ChatCompletionsClient
from chrys.service.llm.openai_exceptions import (
    ContentFilterCodes,
    ContentFilterResultSeverity,
    OpenAIContentFilterException,
)
from chrys.service.llm.openai_responses.client import ResponsesApiClient

_MISSING = object()


def _rejection(inner_code: object) -> dict[str, Any]:
    inner: dict[str, Any] = {
        "content_filter_result": {
            "hate": {"filtered": True, "severity": "high"},
            "violence": {"filtered": False, "severity": None},
            "self_harm": {"filtered": False, "severity": "extreme"},
        }
    }
    if inner_code is not _MISSING:
        inner["code"] = inner_code
    return {
        "error": {
            "message": "The response was filtered.",
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "innererror": inner,
        }
    }


@pytest.mark.parametrize("client_type", [ChatCompletionsClient, ResponsesApiClient])
@pytest.mark.parametrize(
    ("inner_code", "expected"),
    [
        ("ResponsibleAIPolicyViolation", ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION),
        ("ContentFiltered", ContentFilterCodes.CONTENT_FILTERED),
        (_MISSING, ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION),
        ("SomeFutureCode", ContentFilterCodes.UNKNOWN),
        (None, ContentFilterCodes.UNKNOWN),
    ],
)
async def test_a_content_filter_rejection_reports_its_code_and_severities(
    client_type: type[ChatCompletionsClient | ResponsesApiClient], inner_code: object, expected: ContentFilterCodes
) -> None:
    body = _rejection(inner_code)
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json=body, request=request))
    async with httpx.AsyncClient(transport=transport) as http_client:
        sdk = AsyncOpenAI(api_key="sk-test", base_url="https://api.test/v1", max_retries=0, http_client=http_client)
        client = client_type(model="test", sdk_client=sdk)
        with pytest.raises(OpenAIContentFilterException) as raised:
            await client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=False)

    assert raised.value.content_filter_code is expected
    assert raised.value.param == "prompt"
    assert {name: result.severity for name, result in raised.value.content_filter_result.items()} == {
        "hate": ContentFilterResultSeverity.HIGH,
        "violence": ContentFilterResultSeverity.UNKNOWN,
        "self_harm": ContentFilterResultSeverity.UNKNOWN,
    }

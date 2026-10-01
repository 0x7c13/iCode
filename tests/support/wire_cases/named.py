# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cases beyond the grid: tuned per-call options, and the two Responses paths that open their SDK stream differently."""

from __future__ import annotations

import json
from typing import Any

from chrys.service.profiles.models.schema import API_STYLE_RESPONSES

from ._kit import (
    WEATHER_REPORT_JSON,
    Case,
    WeatherReport,
    cc_replies,
    cc_weather_turns,
    lookup_tool,
    resp_message,
    resp_replies,
    resp_response,
    weather_question,
)

CONTINUATION = {"response_id": "resp_background_1"}


def _text(text: str = "Sunny, 21 degrees.", *, response_id: str = "resp_golden_1") -> dict[str, Any]:
    return resp_response(response_id=response_id, output=[resp_message("msg_golden_1", text)])


def _with(**options: Any) -> Any:
    return lambda: dict(options)


def _tuned_options() -> dict[str, Any]:
    return {
        "tools": [lookup_tool()],
        "tool_choice": "required",
        "allow_multiple_tool_calls": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": 7,
        "stop": ["END"],
        "user": "golden-user",
        "metadata": {"purpose": "golden"},
        "instructions": "Prefer metric units.",
    }


CASES: dict[str, Case] = {
    # Profile chat options, headers, output cap and base URL, plus tuned per-call options.
    "openai_cc_tuned_send": Case(
        provider="openai",
        base_url="https://llm.example.test/v1",
        profile_fields={
            "chat_options": json.dumps({"frequency_penalty": 0.5, "extra_body": {"enable_thinking": True}}),
            "http_headers": json.dumps({"X-Golden": "yes"}),
            "max_output_tokens": 1024,
        },
        messages=weather_question,
        options=_tuned_options,
        replies=cc_replies(cc_weather_turns(), stream=False),
    ),
    # A Pydantic response_format, which streams through the SDK's text_format helper.
    "openai_responses_structured_stream": Case(
        provider="openai",
        api_style=API_STYLE_RESPONSES,
        stream=True,
        messages=weather_question,
        options=_with(response_format=WeatherReport),
        replies=resp_replies([_text(WEATHER_REPORT_JSON)], stream=True),
    ),
    # A continuation token for a stored background response, which resumes its stream.
    "openai_responses_resume_stream": Case(
        provider="openai",
        api_style=API_STYLE_RESPONSES,
        stream=True,
        messages=weather_question,
        options=_with(store=True, continuation_token=CONTINUATION),
        replies=resp_replies([_text(response_id="resp_background_1")], stream=True),
    ),
}

# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The basic grid: every (provider, api_style) pair, sent and streamed, through one tool round trip."""

from __future__ import annotations

from chrys.service.profiles.models.schema import API_STYLE_RESPONSES

from ._kit import (
    Case,
    anth_replies,
    anth_weather_turns,
    cc_replies,
    cc_weather_turns,
    resp_replies,
    resp_weather_turns,
    weather_question,
    with_lookup,
)

DEEPSEEK_REASONING = {"reasoning_content": "The user wants the weather; call lookup."}
DEEPSEEK_FINAL_REASONING = {"reasoning_content": "The tool answered; summarize."}


def _grid() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for stream in (False, True):
        mode = "stream" if stream else "send"
        reasoning_turns = cc_weather_turns(reasoning=DEEPSEEK_REASONING, final_reasoning=DEEPSEEK_FINAL_REASONING)
        cases[f"openai_cc_{mode}"] = Case(
            provider="openai",
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=cc_replies(cc_weather_turns(), stream=stream),
        )
        cases[f"deepseek_cc_{mode}"] = Case(
            provider="deepseek-openai",
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=cc_replies(reasoning_turns, stream=stream),
        )
        cases[f"glm_cc_{mode}"] = Case(
            provider="glm-openai",
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=cc_replies(reasoning_turns, stream=stream),
        )
        cases[f"openai_responses_{mode}"] = Case(
            provider="openai",
            api_style=API_STYLE_RESPONSES,
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=resp_replies(resp_weather_turns(), stream=stream),
        )
        cases[f"deepseek_responses_{mode}"] = Case(
            provider="deepseek-openai",
            api_style=API_STYLE_RESPONSES,
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=resp_replies(resp_weather_turns(encrypted=False), stream=stream),
        )
        cases[f"anthropic_{mode}"] = Case(
            provider="anthropic",
            stream=stream,
            messages=weather_question,
            options=with_lookup,
            replies=anth_replies(anth_weather_turns(), stream=stream),
        )
    return cases


CASES = _grid()

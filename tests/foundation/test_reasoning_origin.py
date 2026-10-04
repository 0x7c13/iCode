# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reasoning origin stamps: the endpoint they name and where stamped reasoning may replay."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.reasoning_origin import REASONING_ORIGIN_KEY, ReasoningOrigin, replays_to

ANTHROPIC = ReasoningOrigin("anthropic_messages", "https://api.anthropic.com:443")


@pytest.mark.parametrize(
    ("base_url", "origin"),
    [
        ("https://api.anthropic.com", "https://api.anthropic.com:443"),
        ("HTTPS://API.Anthropic.COM:443/v1/", "https://api.anthropic.com:443"),
        ("http://localhost:8080/anthropic", "http://localhost:8080"),
        ("http://[::1]:8080/v1", "http://[::1]:8080"),
    ],
)
def test_an_endpoint_is_its_protocol_and_the_origin_of_its_base_url(base_url: str, origin: str) -> None:
    assert ReasoningOrigin.of("anthropic_messages", base_url) == ReasoningOrigin("anthropic_messages", origin)


@pytest.mark.parametrize("base_url", ["", "api.anthropic.com", "ftp://host/v1", "Unknown"])
def test_a_base_url_without_an_origin_names_no_endpoint(base_url: str) -> None:
    assert ReasoningOrigin.of("anthropic_messages", base_url) is None


def test_stamped_reasoning_replays_only_to_the_endpoint_that_issued_it() -> None:
    properties: dict[str, Any] = {}
    ANTHROPIC.stamp(properties)

    assert replays_to(properties, ReasoningOrigin.of("anthropic_messages", "https://API.anthropic.com:443/"))
    assert not replays_to(properties, ReasoningOrigin("anthropic_messages", "https://open.bigmodel.cn:443"))
    assert not replays_to(properties, ReasoningOrigin("chat_completions", "https://api.anthropic.com:443"))
    assert not replays_to(properties, None)


def test_unstamped_reasoning_replays_anywhere() -> None:
    assert replays_to({}, ANTHROPIC)
    assert replays_to({"openai_reasoning_format": "reasoning_details"}, None)


@pytest.mark.parametrize(
    "stamp",
    [
        None,
        "https://api.anthropic.com:443",
        {"v": 2, "protocol": "anthropic_messages", "origin": "https://api.anthropic.com:443"},
        {"protocol": "anthropic_messages", "origin": "https://api.anthropic.com:443"},
        {"v": 1, "protocol": "anthropic_messages", "origin": "https://api.anthropic.com:443", "tenant": "t"},
    ],
    ids=["null", "string", "unknown-version", "no-version", "extra-field"],
)
def test_a_stamp_that_does_not_read_as_this_version_never_replays(stamp: object) -> None:
    assert not replays_to({REASONING_ORIGIN_KEY: stamp}, ANTHROPIC)


def test_each_stamp_is_a_fresh_dict() -> None:
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    ANTHROPIC.stamp(first)
    ANTHROPIC.stamp(second)

    assert first[REASONING_ORIGIN_KEY] == second[REASONING_ORIGIN_KEY]
    assert first[REASONING_ORIGIN_KEY] is not second[REASONING_ORIGIN_KEY]

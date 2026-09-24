# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for provider-neutral hosted-tool vocabulary."""

from __future__ import annotations

import pytest

from chrys.foundation.hosted_tools import (
    HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY,
    HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)


@pytest.mark.parametrize(
    ("provider_status", "expected"),
    [
        (None, HostedToolStatus.UNKNOWN),
        ("queued", HostedToolStatus.PENDING),
        ("in_progress", HostedToolStatus.RUNNING),
        ("searching", HostedToolStatus.RUNNING),
        ("interpreting", HostedToolStatus.RUNNING),
        ("generating", HostedToolStatus.RUNNING),
        ("completed", HostedToolStatus.COMPLETED),
        ("incomplete", HostedToolStatus.FAILED),
        ("failed", HostedToolStatus.FAILED),
        ("cancelled", HostedToolStatus.INTERRUPTED),
        ("provider-specific", HostedToolStatus.UNKNOWN),
    ],
)
def test_normalize_hosted_tool_status(provider_status: str | None, expected: HostedToolStatus) -> None:
    assert normalize_hosted_tool_status(provider_status) is expected


def test_hosted_tool_family_defaults_cover_the_vocabulary() -> None:
    families = set(HostedToolFamily)

    assert set(HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY) == families
    assert set(HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY) == families
    assert HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY[HostedToolFamily.SEARCH] == "search"
    assert HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY[HostedToolFamily.MCP] == "mcp"
    assert HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY[HostedToolFamily.SHELL] == "shell"
    assert HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY[HostedToolFamily.GENERIC] == "Hosted Tool"


def test_hosted_tool_vocabulary_values_are_wire_strings() -> None:
    assert HostedToolPhase.TERMINAL == "terminal"
    assert HostedRetrySafety.SIDE_EFFECTFUL == "side_effectful"
